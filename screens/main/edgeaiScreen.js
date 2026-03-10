import { Buffer } from "buffer";
import { CameraView, useCameraPermissions } from "expo-camera";
import * as FileSystem from "expo-file-system/legacy";
import { manipulateAsync, SaveFormat } from "expo-image-manipulator";
import * as jpeg from "jpeg-js";
import { initLlama } from "llama.rn";
import { useEffect, useRef, useState } from "react";
import {
  ActivityIndicator,
  Alert,
  Animated,
  Easing,
  ImageBackground,
  KeyboardAvoidingView,
  Linking,
  Modal,
  Platform,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from "react-native";
import { loadTensorflowModel } from "react-native-fast-tflite";
import { SafeAreaView } from "react-native-safe-area-context";
import { API_BASE_URL } from "../../constants";

// Updated URLs
const BRAIN_URL = `${API_BASE_URL}/edge/download/gemma-3-270m-it.Q4_K_M.gguf`;
const TFLITE_URL = `${API_BASE_URL}/edge/download/yolo11s.tflite`;
const LABELS_URL = `${API_BASE_URL}/edge/download/pests.txt`;

export default function AgriAiScreen() {
  const [permission, requestPermission] = useCameraPermissions();

  // AI Engines
  const [llmContext, setLlmContext] = useState(null);
  const [visionModel, setVisionModel] = useState(null);
  const [pestLabels, setPestLabels] = useState([]); // Updated to pests

  // UI States
  const [appState, setAppState] = useState("initializing");
  const [statusText, setStatusText] = useState("Checking AI models...");
  const [downloadProgress, setDownloadProgress] = useState(0);
  const [userPrompt, setUserPrompt] = useState("");
  const [capturedImage, setCapturedImage] = useState(null);
  const [recommendationResult, setRecommendationResult] = useState(null);

  const cameraRef = useRef(null);
  const analysisProgress = useRef(new Animated.Value(0)).current;

  const cleanPath = (uri) => uri.replace("file://", "");

  const handleRequestPermission = async () => {
    // 1. Actively ask/check the OS for the current status
    const result = await requestPermission();

    // 2. If it's still denied AND the OS won't let us show the native popup anymore
    if (!result.granted && !result.canAskAgain) {
      Alert.alert(
        "Camera Access Required",
        "Camera access is permanently blocked. Please enable it in your device settings to scan crops.",
        [
          { text: "Cancel", style: "cancel" },
          { text: "Open Settings", onPress: () => Linking.openSettings() },
        ],
      );
    }
  };

  useEffect(() => {
    const setup = async () => {
      try {
        const brainPath = `${FileSystem.documentDirectory}gemma3.gguf`;
        const tflitePath = `${FileSystem.documentDirectory}vision.tflite`;
        const labelsPath = `${FileSystem.documentDirectory}labels.txt`;

        // --- 1. Download Gemma (LLM Reasoner) ---
        const brainInfo = await FileSystem.getInfoAsync(brainPath);
        if (!brainInfo.exists || brainInfo.size < 100000000) {
          setAppState("downloading");
          setStatusText("Downloading Gemma-3 Brain (200MB)...");

          const brainResumable = FileSystem.createDownloadResumable(
            BRAIN_URL,
            brainPath,
            {},
            (dp) => {
              setDownloadProgress(
                (dp.totalBytesWritten / dp.totalBytesExpectedToWrite) * 0.8,
              );
            },
          );
          const result = await brainResumable.downloadAsync();
          if (result.status !== 200) throw new Error("Brain Download Failed");
        } else {
          setDownloadProgress(0.8);
        }

        // --- 2. Download TFLite & Labels (Vision) ---
        const tfliteInfo = await FileSystem.getInfoAsync(tflitePath);
        const labelsInfo = await FileSystem.getInfoAsync(labelsPath);

        if (!tfliteInfo.exists || !labelsInfo.exists) {
          setStatusText("Downloading Vision Model & Labels...");
          await FileSystem.downloadAsync(TFLITE_URL, tflitePath);
          await FileSystem.downloadAsync(LABELS_URL, labelsPath);
        }
        setDownloadProgress(1.0);

        // --- 3. Initialize Both Engines ---
        setAppState("initializing");
        setStatusText("Waking up AI Engines...");

        // A. Load Pests Text File into Array
        const labelsText = await FileSystem.readAsStringAsync(labelsPath);
        const loadedLabels = labelsText
          .split("\n")
          .map((l) => l.trim())
          .filter((l) => l !== "");
        setPestLabels(loadedLabels);

        // B. Load YOLO TFLite Model
        console.log("➡️ Attempting to load YOLO Vision...");
        const tflitePlugin = await loadTensorflowModel({ url: tflitePath });
        setVisionModel(tflitePlugin);
        console.log("✅ YOLO Vision Loaded Successfully!");

        // C. Load Gemma LLM
        console.log("➡️ Attempting to load Gemma LLM...");
        const ctx = await initLlama({
          model: cleanPath(brainPath),
          n_gpu_layers: 0, // Keeping CPU mode for mobile stability
          n_ctx: 1024,
        });
        setLlmContext(ctx);
        console.log("✅ Gemma LLM Loaded Successfully!");

        setAppState("ready");
      } catch (error) {
        console.error("Setup Error:", error);
        setStatusText("Failed to load AI.");
      }
    };
    setup();
  }, []);

  // --- YOLO11 IMAGE PROCESSING LOGIC ---
  const formatImageForYOLO = async (uri) => {
    // 1. YOLO models typically expect 640x640 resolution
    const resized = await manipulateAsync(
      uri,
      [{ resize: { width: 640, height: 640 } }],
      { format: SaveFormat.JPEG, base64: true },
    );

    // 2. Decode JPEG to raw pixels using jpeg-js
    const rawImageData = jpeg.decode(Buffer.from(resized.base64, "base64"), {
      useTArray: true,
    });

    // 3. Convert to Float32Array & Normalize (0.0 to 1.0)
    const float32Data = new Float32Array(3 * 640 * 640);
    let offset = 0;
    for (let i = 0; i < rawImageData.data.length; i += 4) {
      float32Data[offset++] = rawImageData.data[i] / 255.0; // R
      float32Data[offset++] = rawImageData.data[i + 1] / 255.0; // G
      float32Data[offset++] = rawImageData.data[i + 2] / 255.0; // B
    }
    return float32Data;
  };

  // --- YOLO TENSOR EXTRACTOR ---
  const extractBestYoloDetection = (outputArray, numClasses) => {
    // YOLO11 TFLite output is typically [1, 4 + numClasses, 8400]
    // This means the first 4 rows are box coordinates, and the rest are class scores.
    const numAnchors = 8400;
    const rowOffset = 4; // Skip x, y, w, h

    let bestScore = 0;
    let bestClassIndex = -1;

    // We need to iterate through classes first, then anchors,
    // because the data is 'Transposed' (packed by attribute, not by anchor)
    for (let c = 0; c < numClasses; c++) {
      const classRowStart = (c + rowOffset) * numAnchors;

      for (let a = 0; a < numAnchors; a++) {
        const score = outputArray[classRowStart + a];

        if (score > bestScore) {
          bestScore = score;
          bestClassIndex = c;
        }
      }
    }

    // Sigmoid check: YOLO raw outputs are sometimes logits.
    // If scores are still > 1, we apply a sigmoid to squash it to 0-1.
    if (bestScore > 1) {
      bestScore = 1 / (1 + Math.exp(-bestScore));
    }

    return { bestClassIndex, bestScore };
  };

  const onCapture = async () => {
    if (!cameraRef.current) return;
    const photo = await cameraRef.current.takePictureAsync({ base64: false });
    setCapturedImage(photo.uri);
    setAppState("preview");
  };

  const onAnalyze = async () => {
    if (
      !llmContext ||
      !visionModel ||
      !capturedImage ||
      pestLabels.length === 0
    )
      return;

    setAppState("analyzing");

    analysisProgress.setValue(0);
    Animated.timing(analysisProgress, {
      toValue: 0.85,
      duration: 8000,
      easing: Easing.out(Easing.cubic),
      useNativeDriver: false,
    }).start();

    try {
      // --- STEP 1: VISION (YOLO11s) ---
      setStatusText("Scanning for pests...");
      const inputTensor = await formatImageForYOLO(capturedImage);
      const output = await visionModel.run([inputTensor]);

      const { bestClassIndex, bestScore } = extractBestYoloDetection(
        output[0],
        pestLabels.length,
      );

      // Filter out low-confidence hallucinations
      if (bestScore < 0.25 || bestClassIndex === -1) {
        Animated.timing(analysisProgress, {
          toValue: 1,
          duration: 400,
          useNativeDriver: false,
        }).start(() => {
          Alert.alert(
            "Scan Complete",
            "No major pests detected with high confidence.",
          );
          setCapturedImage(null);
          setUserPrompt("");
          setAppState("ready");
        });
        return;
      }

      const detectedPest = pestLabels[bestClassIndex];

      // --- STEP 2: REASONING (Gemma-3) ---
      const displayScore = (bestScore * 100).toFixed(1);
      setStatusText(`Found: ${detectedPest} (${displayScore}%). Consulting...`);

      const userText =
        userPrompt.trim() !== ""
          ? userPrompt
          : "Give 3 organic ways to safely treat this pest.";

      // Format exactly for Gemma Instruction Models
      const gemmaPrompt = `<start_of_turn>user\nA farmer found a ${detectedPest} infestation on their crops. ${userText}<end_of_turn>\n<start_of_turn>model\n`;

      const result = await llmContext.completion({
        prompt: gemmaPrompt,
        max_tokens: 400,
        temperature: 0.1, // Highly factual
      });

      Animated.timing(analysisProgress, {
        toValue: 1,
        duration: 400,
        useNativeDriver: false,
      }).start(() => {
        // Construct the final beautiful output
        const finalOutput = `🐞 Pest Detected: ${detectedPest.toUpperCase()}\nConfidence: ${Math.round(bestScore * 100)}%\n\n${result.text.trim()}`;
        setRecommendationResult(finalOutput);

        setUserPrompt("");
        setCapturedImage(null);
        setAppState("ready");
      });
    } catch (e) {
      console.error("Inference Error:", e);
      Alert.alert("Error", "Failed to analyze the image.");
      setAppState("preview");
    }
  };

  const onRetake = () => {
    setCapturedImage(null);
    setUserPrompt("");
    setAppState("ready");
  };

  // --- RENDERS ---
  if (!permission?.granted) {
    return (
      <SafeAreaView style={styles.centerContainer}>
        <View style={styles.permissionCard}>
          <View style={styles.permissionIcon}>
            <Text style={{ fontSize: 36 }}>📸</Text>
          </View>
          <Text style={styles.permissionTitle}>Camera Access Required</Text>
          <Text style={styles.permissionDescription}>
            Agri-AI needs camera access to scan crops and detect pests. Please
            grant permission to continue.
          </Text>
          <TouchableOpacity
            style={styles.permissionButton}
            onPress={handleRequestPermission}
            activeOpacity={0.8}
          >
            <Text style={styles.permissionButtonText}>Grant Camera Access</Text>
          </TouchableOpacity>
        </View>
      </SafeAreaView>
    );
  }

  if (appState === "initializing" || appState === "downloading") {
    return (
      <SafeAreaView style={styles.setupContainer}>
        <View style={styles.setupContent}>
          <ActivityIndicator
            size="large"
            color="#4CAF50"
            style={{ marginBottom: 20 }}
          />
          <Text style={styles.setupTitle}>Agri-AI Edge</Text>
          <Text style={styles.setupSubtitle}>{statusText}</Text>
          {appState === "downloading" && (
            <View style={styles.progressBarContainer}>
              <View
                style={[
                  styles.progressBarFill,
                  { width: `${downloadProgress * 100}%` },
                ]}
              />
              <Text style={styles.progressText}>
                {Math.round(downloadProgress * 100)}%
              </Text>
            </View>
          )}
        </View>
      </SafeAreaView>
    );
  }

  if (appState === "preview" || appState === "analyzing") {
    return (
      <KeyboardAvoidingView
        style={{ flex: 1 }}
        behavior={Platform.OS === "ios" ? "padding" : "height"}
      >
        <ImageBackground source={{ uri: capturedImage }} style={{ flex: 1 }}>
          <SafeAreaView style={styles.cameraOverlay}>
            <View style={styles.previewHeader}>
              <TouchableOpacity
                onPress={onRetake}
                disabled={appState === "analyzing"}
              >
                <Text style={styles.retakeText}>✕ Retake</Text>
              </TouchableOpacity>
            </View>
            <View style={styles.previewFooter}>
              <TextInput
                style={styles.textInput}
                placeholder="Ask a question (optional)..."
                placeholderTextColor="#A5D6A7"
                value={userPrompt}
                onChangeText={setUserPrompt}
                editable={appState !== "analyzing"}
              />
              <TouchableOpacity
                onPress={onAnalyze}
                style={styles.analyzeSubmitButton}
                disabled={appState === "analyzing"}
              >
                <Text style={styles.analyzeSubmitText}>Analyze Crop</Text>
              </TouchableOpacity>
            </View>
          </SafeAreaView>

          {appState === "analyzing" && (
            <View style={styles.analyzingOverlay}>
              <Text style={styles.analyzingText}>{statusText}</Text>
              <View style={styles.horizontalBarContainer}>
                <Animated.View
                  style={[
                    styles.horizontalBarFill,
                    {
                      width: analysisProgress.interpolate({
                        inputRange: [0, 1],
                        outputRange: ["0%", "100%"],
                      }),
                    },
                  ]}
                />
              </View>
            </View>
          )}
        </ImageBackground>
      </KeyboardAvoidingView>
    );
  }

  return (
    <View style={{ flex: 1 }}>
      <CameraView style={{ flex: 1 }} ref={cameraRef} facing="back">
        <SafeAreaView style={styles.cameraOverlay}>
          <View style={styles.header}>
            <Text style={styles.headerText}>Point at the affected crop</Text>
          </View>
          <View style={styles.viewfinderFrame} />
          <View style={styles.footer}>
            <TouchableOpacity onPress={onCapture} style={styles.captureButton}>
              <View style={styles.captureButtonInner} />
            </TouchableOpacity>
          </View>
        </SafeAreaView>
      </CameraView>

      <Modal
        animationType="slide"
        transparent={true}
        visible={recommendationResult !== null}
        onRequestClose={() => setRecommendationResult(null)}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <View style={styles.modalHandle} />
            <Text style={styles.modalTitle}>Diagnosis & Recommendation</Text>
            <ScrollView style={styles.modalScroll}>
              <Text style={styles.modalText}>{recommendationResult}</Text>
            </ScrollView>
            <TouchableOpacity
              style={styles.modalCloseButton}
              onPress={() => setRecommendationResult(null)}
            >
              <Text style={styles.modalCloseButtonText}>Done</Text>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>
    </View>
  );
}

// Ensure you keep your existing styles block here at the bottom
const styles = StyleSheet.create({
  centerContainer: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    backgroundColor: "#1E1E1E",
    padding: 24,
  },
  setupContainer: {
    flex: 1,
    backgroundColor: "#1E1E1E",
    justifyContent: "center",
    alignItems: "center",
  },
  setupContent: { width: "80%", alignItems: "center" },
  setupTitle: {
    fontSize: 22,
    fontWeight: "bold",
    color: "#ffffff",
    marginBottom: 8,
  },
  setupSubtitle: {
    fontSize: 14,
    color: "#A5D6A7",
    marginBottom: 30,
    textAlign: "center",
  },
  progressBarContainer: {
    width: "100%",
    height: 12,
    backgroundColor: "#333333",
    borderRadius: 6,
    overflow: "hidden",
  },
  progressBarFill: { height: "100%", backgroundColor: "#4CAF50" },
  progressText: {
    color: "#ffffff",
    marginTop: 10,
    fontSize: 14,
    fontWeight: "bold",
  },
  cameraOverlay: { flex: 1, justifyContent: "space-between" },
  header: {
    backgroundColor: "rgba(0,0,0,0.5)",
    padding: 15,
    alignItems: "center",
  },
  headerText: { color: "white", fontSize: 16, fontWeight: "600" },
  viewfinderFrame: {
    width: 250,
    height: 250,
    borderWidth: 2,
    borderColor: "rgba(76, 175, 80, 0.7)",
    alignSelf: "center",
    borderRadius: 12,
  },
  footer: { paddingBottom: 40, alignItems: "center" },
  captureButton: {
    width: 70,
    height: 70,
    borderRadius: 35,
    backgroundColor: "rgba(255,255,255,0.3)",
    justifyContent: "center",
    alignItems: "center",
  },
  captureButtonInner: {
    width: 54,
    height: 54,
    borderRadius: 27,
    backgroundColor: "white",
  },
  previewHeader: { padding: 20, alignItems: "flex-start" },
  retakeText: {
    color: "white",
    fontSize: 18,
    fontWeight: "bold",
    textShadowColor: "rgba(0, 0, 0, 0.75)",
    textShadowOffset: { width: -1, height: 1 },
    textShadowRadius: 10,
  },
  previewFooter: {
    padding: 20,
    backgroundColor: "rgba(0,0,0,0.6)",
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
  },
  textInput: {
    backgroundColor: "rgba(255, 255, 255, 0.1)",
    color: "white",
    padding: 15,
    borderRadius: 12,
    fontSize: 16,
    marginBottom: 15,
    borderWidth: 1,
    borderColor: "rgba(76, 175, 80, 0.5)",
    minHeight: 50,
  },
  analyzeSubmitButton: {
    backgroundColor: "#4CAF50",
    padding: 18,
    borderRadius: 12,
    alignItems: "center",
  },
  analyzeSubmitText: { color: "white", fontWeight: "bold", fontSize: 18 },
  analyzingOverlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: "rgba(0, 0, 0, 0.8)",
    justifyContent: "center",
    alignItems: "center",
    zIndex: 10,
  },
  analyzingText: {
    color: "white",
    fontSize: 20,
    fontWeight: "bold",
    marginTop: 20,
  },
  horizontalBarContainer: {
    width: "80%",
    height: 8,
    backgroundColor: "rgba(255, 255, 255, 0.2)",
    borderRadius: 4,
    marginTop: 20,
    overflow: "hidden",
  },
  horizontalBarFill: {
    height: "100%",
    backgroundColor: "#4CAF50",
    borderRadius: 4,
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: "rgba(0, 0, 0, 0.5)",
    justifyContent: "flex-end",
  },
  modalContent: {
    backgroundColor: "#1E1E1E",
    borderTopLeftRadius: 24,
    borderTopRightRadius: 24,
    padding: 24,
    maxHeight: "80%",
  },
  modalHandle: {
    width: 40,
    height: 5,
    backgroundColor: "#555",
    borderRadius: 3,
    alignSelf: "center",
    marginBottom: 20,
  },
  modalTitle: {
    fontSize: 22,
    fontWeight: "bold",
    color: "#4CAF50",
    marginBottom: 15,
  },
  modalScroll: { marginBottom: 20 },
  modalText: { fontSize: 16, color: "#E0E0E0", lineHeight: 24 },
  modalCloseButton: {
    backgroundColor: "#4CAF50",
    paddingVertical: 15,
    borderRadius: 12,
    alignItems: "center",
  },
  modalCloseButtonText: { color: "white", fontSize: 16, fontWeight: "bold" },
  permissionCard: {
    backgroundColor: "#2A2A2A",
    borderRadius: 20,
    padding: 32,
    width: "100%",
    maxWidth: 400,
    alignItems: "center",
    borderWidth: 1,
    borderColor: "#4CAF50",
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 8,
  },

  permissionIcon: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: "rgba(76, 175, 80, 0.15)",
    justifyContent: "center",
    alignItems: "center",
    marginBottom: 24,
  },

  permissionTitle: {
    fontSize: 24,
    fontWeight: "bold",
    color: "#FFFFFF",
    marginBottom: 12,
    textAlign: "center",
  },

  permissionDescription: {
    fontSize: 16,
    color: "#A5A5A5",
    textAlign: "center",
    marginBottom: 28,
    lineHeight: 22,
  },

  permissionButton: {
    backgroundColor: "#4CAF50",
    paddingVertical: 16,
    paddingHorizontal: 32,
    borderRadius: 14,
    width: "100%",
    alignItems: "center",
    shadowColor: "#4CAF50",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 4,
    elevation: 3,
  },

  permissionButtonText: {
    color: "#FFFFFF",
    fontSize: 18,
    fontWeight: "bold",
    letterSpacing: 0.5,
  },
});
