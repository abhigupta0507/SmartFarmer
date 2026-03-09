import React, {
  useState,
  useRef,
  useEffect,
  useCallback,
  useMemo,
} from 'react';
import {
  View,
  Text,
  StyleSheet,
  FlatList,
  TextInput,
  TouchableOpacity,
  KeyboardAvoidingView,
  Platform,
  SafeAreaView,
  ActivityIndicator,
  Animated,
  Vibration,
  Alert,
} from 'react-native';
import { Audio } from 'expo-av';
import { useAuth } from '../../context/AuthContext';
import chatService from '../../services/chat.service';
import { COLORS } from '../../constants';

// ─── Constants ───────────────────────────────────────────────────────────────

const SUGGESTED_PROMPTS = [
  { id: 1, text: '🌾 मेरी फसल पर कीड़े लग गए हैं', label: 'कीट समस्या' },
  { id: 2, text: '💧 सिंचाई कब करनी चाहिए?', label: 'सिंचाई' },
  { id: 3, text: '🌱 गेहूं में कौन सी खाद डालें?', label: 'खाद' },
  { id: 4, text: '☁️ आज मौसम कैसा रहेगा?', label: 'मौसम' },
];

// ─── Message Bubble ───────────────────────────────────────────────────────────

const MessageBubble = React.memo(({ message }) => {
  const isUser = message.role === 'user';
  const fadeAnim = useRef(new Animated.Value(0)).current;
  const slideAnim = useRef(new Animated.Value(isUser ? 20 : -20)).current;

  useEffect(() => {
    Animated.parallel([
      Animated.timing(fadeAnim, {
        toValue: 1,
        duration: 280,
        useNativeDriver: true,
      }),
      Animated.spring(slideAnim, {
        toValue: 0,
        tension: 60,
        friction: 8,
        useNativeDriver: true,
      }),
    ]).start();
  }, []);

  return (
    <Animated.View
      style={[
        styles.bubbleRow,
        isUser ? styles.bubbleRowUser : styles.bubbleRowAI,
        { opacity: fadeAnim, translateX: slideAnim },
      ]}
    >
      {!isUser && (
        <View style={styles.aiAvatar}>
          <Text style={styles.aiAvatarText}>🌾</Text>
        </View>
      )}
      <View
        style={[
          styles.bubble,
          isUser ? styles.bubbleUser : styles.bubbleAI,
          message.isVoice && styles.bubbleVoice,
        ]}
      >
        {message.isVoice && (
          <Text style={styles.voiceTag}>
            {isUser ? '🎙️ Voice' : '🔊 Voice Reply'}
          </Text>
        )}
        <Text style={[styles.bubbleText, isUser ? styles.bubbleTextUser : styles.bubbleTextAI]}>
          {message.content}
        </Text>
        <Text style={[styles.timestamp, isUser ? styles.timestampUser : styles.timestampAI]}>
          {message.time}
        </Text>
      </View>
    </Animated.View>
  );
});

// ─── Typing Indicator ─────────────────────────────────────────────────────────

const TypingIndicator = () => {
  const dot1 = useRef(new Animated.Value(0)).current;
  const dot2 = useRef(new Animated.Value(0)).current;
  const dot3 = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    const animate = (dot, delay) =>
      Animated.loop(
        Animated.sequence([
          Animated.delay(delay),
          Animated.timing(dot, { toValue: -6, duration: 300, useNativeDriver: true }),
          Animated.timing(dot, { toValue: 0, duration: 300, useNativeDriver: true }),
          Animated.delay(600),
        ])
      ).start();

    animate(dot1, 0);
    animate(dot2, 150);
    animate(dot3, 300);
  }, []);

  return (
    <View style={styles.bubbleRow}>
      <View style={styles.aiAvatar}>
        <Text style={styles.aiAvatarText}>🌾</Text>
      </View>
      <View style={[styles.bubble, styles.bubbleAI, styles.typingBubble]}>
        {[dot1, dot2, dot3].map((dot, i) => (
          <Animated.View
            key={i}
            style={[styles.typingDot, { transform: [{ translateY: dot }] }]}
          />
        ))}
      </View>
    </View>
  );
};

// ─── Voice Recording Button ───────────────────────────────────────────────────

const VoiceButton = ({ onPress, isRecording, isProcessing }) => {
  const pulseAnim = useRef(new Animated.Value(1)).current;
  const ringAnim = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    if (isRecording) {
      Animated.loop(
        Animated.sequence([
          Animated.timing(pulseAnim, { toValue: 1.12, duration: 600, useNativeDriver: true }),
          Animated.timing(pulseAnim, { toValue: 1, duration: 600, useNativeDriver: true }),
        ])
      ).start();
      Animated.loop(
        Animated.timing(ringAnim, { toValue: 1, duration: 1200, useNativeDriver: true })
      ).start();
    } else {
      pulseAnim.setValue(1);
      ringAnim.setValue(0);
    }
  }, [isRecording]);

  const ringScale = ringAnim.interpolate({ inputRange: [0, 1], outputRange: [1, 1.6] });
  const ringOpacity = ringAnim.interpolate({ inputRange: [0, 0.5, 1], outputRange: [0.6, 0.2, 0] });

  return (
    <TouchableOpacity
      onPress={onPress}
      disabled={isProcessing}
      activeOpacity={0.85}
      style={styles.voiceBtnWrapper}
    >
      {isRecording && (
        <Animated.View
          style={[
            styles.voiceRing,
            { transform: [{ scale: ringScale }], opacity: ringOpacity },
          ]}
        />
      )}
      <Animated.View
        style={[
          styles.voiceBtn,
          isRecording && styles.voiceBtnActive,
          isProcessing && styles.voiceBtnProcessing,
          { transform: [{ scale: pulseAnim }] },
        ]}
      >
        {isProcessing ? (
          <ActivityIndicator color="#fff" size="small" />
        ) : (
          <Text style={styles.voiceBtnIcon}>{isRecording ? '⏹️' : '🎙️'}</Text>
        )}
      </Animated.View>
    </TouchableOpacity>
  );
};

// ─── Main Screen ──────────────────────────────────────────────────────────────

const ChatScreen = () => {
  const { user } = useAuth();
  const [messages, setMessages] = useState([]);
  const [inputText, setInputText] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessingVoice, setIsProcessingVoice] = useState(false);
  const [showSuggestions, setShowSuggestions] = useState(true);

  const flatListRef = useRef(null);
  const recordingRef = useRef(null);
  const soundRef = useRef(null);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (recordingRef.current) {
        recordingRef.current.stopAndUnloadAsync().catch(() => {});
      }
      if (soundRef.current) {
        soundRef.current.unloadAsync().catch(() => {});
      }
    };
  }, []);

  const getTime = () => {
    const now = new Date();
    return `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}`;
  };

  // Build conversation history in the format the API expects
  const buildHistory = useCallback(() => {
    return messages.map((m) => ({
      role: m.role,
      content: m.content,
    }));
  }, [messages]);

  const addMessage = useCallback((role, content, isVoice = false) => {
    const msg = {
      id: `${Date.now()}-${Math.random()}`,
      role,
      content,
      time: getTime(),
      isVoice,
    };
    setMessages((prev) => [...prev, msg]);
    return msg;
  }, []);

  const scrollToBottom = useCallback(() => {
    setTimeout(() => {
      flatListRef.current?.scrollToEnd({ animated: true });
    }, 100);
  }, []);

  // ── Text Send ─────────────────────────────────────────────────────────────

  const handleSend = useCallback(async () => {
    const text = inputText.trim();
    if (!text || isTyping) return;

    setInputText('');
    setShowSuggestions(false);
    addMessage('user', text);
    setIsTyping(true);
    scrollToBottom();

    try {
      // Build history BEFORE adding current message (addMessage is async state update)
      const history = buildHistory();
      const allMessages = [...history, { role: 'user', content: text }];
      const reply = await chatService.sendMessage(allMessages);
      addMessage('assistant', reply);
    } catch (err) {
      addMessage('assistant', 'माफ़ करें, कुछ गड़बड़ हो गई। कृपया दोबारा कोशिश करें। 🙏');
    } finally {
      setIsTyping(false);
      scrollToBottom();
    }
  }, [inputText, isTyping, buildHistory, addMessage, scrollToBottom]);

  // ── Suggestion Tap ────────────────────────────────────────────────────────

  const handleSuggestion = useCallback((text) => {
    setInputText(text);
  }, []);

  // ── Voice Recording ───────────────────────────────────────────────────────

  const startRecording = useCallback(async () => {
    try {
      const { status } = await Audio.requestPermissionsAsync();
      if (status !== 'granted') {
        Alert.alert('Permission Required', 'Microphone access is needed for voice messages.');
        return;
      }

      await Audio.setAudioModeAsync({
        allowsRecordingIOS: true,
        playsInSilentModeIOS: true,
      });

      const recording = new Audio.Recording();
      await recording.prepareToRecordAsync(
        Audio.RecordingOptionsPresets.HIGH_QUALITY
      );
      await recording.startAsync();

      recordingRef.current = recording;
      setIsRecording(true);
      Vibration.vibrate(50);
    } catch (err) {
      console.error('Recording start error:', err);
      Alert.alert('Error', 'Could not start recording. Please try again.');
    }
  }, []);

  const stopRecording = useCallback(async () => {
    if (!recordingRef.current) return;

    setIsRecording(false);
    setIsProcessingVoice(true);
    Vibration.vibrate(50);

    try {
      await recordingRef.current.stopAndUnloadAsync();
      const uri = recordingRef.current.getURI();
      recordingRef.current = null;

      if (!uri) throw new Error('No audio URI');

      setShowSuggestions(false);

      // Get history BEFORE adding user message
      const history = buildHistory();

      const { userText, replyText, audioBlob } = await chatService.sendVoiceMessage(uri, history);

      // Add user's transcribed speech to chat
      if (userText) addMessage('user', userText, true);

      // Add AI reply to chat
      if (replyText) addMessage('assistant', replyText, true);

      // Play audio response
      if (audioBlob) {
        await playAudioBlob(audioBlob);
      }

      scrollToBottom();
    } catch (err) {
      console.error('Voice processing error:', err);
      addMessage('assistant', 'आवाज़ नहीं सुन पाया। कृपया दोबारा बोलें या टाइप करें। 🎙️');
      scrollToBottom();
    } finally {
      setIsProcessingVoice(false);
    }
  }, [buildHistory, addMessage, scrollToBottom]);

  const handleVoicePress = useCallback(() => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  }, [isRecording, startRecording, stopRecording]);

  // ── Audio Playback ────────────────────────────────────────────────────────

  const playAudioBlob = async (blob) => {
    try {
      // Unload previous sound
      if (soundRef.current) {
        await soundRef.current.unloadAsync();
        soundRef.current = null;
      }

      // Convert blob to base64 data URI for expo-av
      const reader = new FileReader();
      const base64 = await new Promise((resolve, reject) => {
        reader.onloadend = () => resolve(reader.result);
        reader.onerror = reject;
        reader.readAsDataURL(blob);
      });

      await Audio.setAudioModeAsync({
        allowsRecordingIOS: false,
        playsInSilentModeIOS: true,
        shouldDuckAndroid: true,
        playThroughEarpieceAndroid: false,
      });

      const { sound } = await Audio.Sound.createAsync({ uri: base64 });
      soundRef.current = sound;
      await sound.playAsync();
    } catch (err) {
      console.error('Audio playback error:', err);
      // Non-fatal — the text reply is already shown in chat
    }
  };

  // ── Welcome Message ───────────────────────────────────────────────────────

  const WelcomeHeader = useMemo(
    () => (
      <View style={styles.welcomeContainer}>
        <View style={styles.welcomeAvatar}>
          <Text style={styles.welcomeAvatarEmoji}>🌾</Text>
        </View>
        <Text style={styles.welcomeTitle}>Krishi AI</Text>
        <Text style={styles.welcomeSubtitle}>
          नमस्ते {user?.fullName?.split(' ')[0] || 'किसान भाई'}! 👋{'\n'}
          आपकी खेती में मैं कैसे मदद कर सकता हूँ?
        </Text>
        <View style={styles.welcomeFeatures}>
          {['🌱 फसल सलाह', '🐛 कीट पहचान', '💧 सिंचाई', '📈 मंडी भाव'].map((f) => (
            <View key={f} style={styles.featureTag}>
              <Text style={styles.featureTagText}>{f}</Text>
            </View>
          ))}
        </View>
      </View>
    ),
    [user]
  );

  const renderItem = useCallback(
    ({ item }) => <MessageBubble message={item} />,
    []
  );

  const keyExtractor = useCallback((item) => item.id, []);

  return (
    <SafeAreaView style={styles.safe}>
      {/* Header */}
      <View style={styles.header}>
        <View style={styles.headerLeft}>
          <View style={styles.headerAvatar}>
            <Text style={styles.headerAvatarText}>🌾</Text>
          </View>
          <View>
            <Text style={styles.headerTitle}>Krishi AI</Text>
            <View style={styles.onlineRow}>
              <View style={styles.onlineDot} />
              <Text style={styles.onlineText}>Online • Hindi & English</Text>
            </View>
          </View>
        </View>
        <TouchableOpacity
          onPress={() => {
            Alert.alert('Clear Chat', 'Start a new conversation?', [
              { text: 'Cancel', style: 'cancel' },
              {
                text: 'Clear',
                style: 'destructive',
                onPress: () => {
                  setMessages([]);
                  setShowSuggestions(true);
                },
              },
            ]);
          }}
          style={styles.clearBtn}
        >
          <Text style={styles.clearBtnText}>🗑️</Text>
        </TouchableOpacity>
      </View>

      <KeyboardAvoidingView
        style={styles.flex}
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        keyboardVerticalOffset={Platform.OS === 'ios' ? 0 : 20}
      >
        {/* Messages */}
        <FlatList
          ref={flatListRef}
          data={messages}
          renderItem={renderItem}
          keyExtractor={keyExtractor}
          contentContainerStyle={styles.messageList}
          ListHeaderComponent={messages.length === 0 ? WelcomeHeader : null}
          ListFooterComponent={isTyping ? <TypingIndicator /> : null}
          onContentSizeChange={scrollToBottom}
          showsVerticalScrollIndicator={false}
          removeClippedSubviews={false}
        />

        {/* Suggestions */}
        {showSuggestions && messages.length === 0 && (
          <View style={styles.suggestionsContainer}>
            <Text style={styles.suggestionsLabel}>अक्सर पूछे जाने वाले सवाल</Text>
            <View style={styles.suggestionChips}>
              {SUGGESTED_PROMPTS.map((s) => (
                <TouchableOpacity
                  key={s.id}
                  style={styles.chip}
                  onPress={() => handleSuggestion(s.text)}
                  activeOpacity={0.75}
                >
                  <Text style={styles.chipText}>{s.label}</Text>
                </TouchableOpacity>
              ))}
            </View>
          </View>
        )}

        {/* Input Bar */}
        <View style={styles.inputBar}>
          {/* Recording indicator */}
          {isRecording && (
            <View style={styles.recordingBanner}>
              <View style={styles.recordingDot} />
              <Text style={styles.recordingText}>Recording... बोलते रहें</Text>
            </View>
          )}

          <View style={styles.inputRow}>
            <TextInput
              style={styles.textInput}
              value={inputText}
              onChangeText={setInputText}
              placeholder="कुछ पूछें... (हिंदी या English)"
              placeholderTextColor={COLORS.textMuted}
              multiline
              maxLength={500}
              onSubmitEditing={handleSend}
              blurOnSubmit={false}
              editable={!isRecording && !isProcessingVoice}
            />

            {inputText.trim().length > 0 ? (
              <TouchableOpacity
                style={styles.sendBtn}
                onPress={handleSend}
                disabled={isTyping}
                activeOpacity={0.85}
              >
                {isTyping ? (
                  <ActivityIndicator color="#fff" size="small" />
                ) : (
                  <Text style={styles.sendBtnIcon}>➤</Text>
                )}
              </TouchableOpacity>
            ) : (
              <VoiceButton
                onPress={handleVoicePress}
                isRecording={isRecording}
                isProcessing={isProcessingVoice}
              />
            )}
          </View>

          <Text style={styles.disclaimer}>
            AI सलाह है — गंभीर समस्या के लिए कृषि विशेषज्ञ से मिलें
          </Text>
        </View>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
};

// ─── Styles ───────────────────────────────────────────────────────────────────

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: '#F5F7F0' },
  flex: { flex: 1 },

  // Header
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingVertical: 12,
    marginTop:25,
    backgroundColor: COLORS.primary,
    borderBottomLeftRadius: 0,
    borderBottomRightRadius: 0,
  },
  headerLeft: { flexDirection: 'row', alignItems: 'center', gap: 10 },
  headerAvatar: {
    width: 42,
    height: 42,
    borderRadius: 21,
    backgroundColor: 'rgba(255,255,255,0.2)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  headerAvatarText: { fontSize: 22 },
  headerTitle: { fontSize: 16, fontWeight: '800', color: '#fff' },
  onlineRow: { flexDirection: 'row', alignItems: 'center', gap: 4, marginTop: 2 },
  onlineDot: {
    width: 7,
    height: 7,
    borderRadius: 3.5,
    backgroundColor: '#A8E6A3',
  },
  onlineText: { fontSize: 11, color: 'rgba(255,255,255,0.8)' },
  clearBtn: { padding: 8 },
  clearBtnText: { fontSize: 18 },

  // Messages
  messageList: {
    paddingHorizontal: 12,
    paddingTop: 12,
    paddingBottom: 8,
    flexGrow: 1,
  },

  // Welcome
  welcomeContainer: {
    alignItems: 'center',
    paddingVertical: 28,
    paddingHorizontal: 20,
  },
  welcomeAvatar: {
    width: 72,
    height: 72,
    borderRadius: 36,
    backgroundColor: COLORS.primary,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 14,
    shadowColor: COLORS.primary,
    shadowOpacity: 0.3,
    shadowRadius: 10,
    shadowOffset: { width: 0, height: 4 },
    elevation: 6,
  },
  welcomeAvatarEmoji: { fontSize: 36 },
  welcomeTitle: {
    fontSize: 22,
    fontWeight: '800',
    color: COLORS.textPrimary,
    marginBottom: 8,
  },
  welcomeSubtitle: {
    fontSize: 14,
    color: COLORS.textSecondary,
    textAlign: 'center',
    lineHeight: 22,
    marginBottom: 18,
  },
  welcomeFeatures: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
    justifyContent: 'center',
  },
  featureTag: {
    backgroundColor: '#E8F5E9',
    borderRadius: 20,
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderWidth: 1,
    borderColor: '#C8E6C9',
  },
  featureTagText: { fontSize: 12, color: COLORS.primaryDark, fontWeight: '600' },

  // Bubbles
  bubbleRow: {
    flexDirection: 'row',
    marginBottom: 10,
    alignItems: 'flex-end',
    gap: 8,
  },
  bubbleRowUser: { justifyContent: 'flex-end' },
  bubbleRowAI: { justifyContent: 'flex-start' },
  aiAvatar: {
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: COLORS.primary,
    alignItems: 'center',
    justifyContent: 'center',
    flexShrink: 0,
  },
  aiAvatarText: { fontSize: 16 },
  bubble: {
    maxWidth: '78%',
    borderRadius: 18,
    paddingHorizontal: 14,
    paddingVertical: 10,
    shadowColor: '#000',
    shadowOpacity: 0.06,
    shadowRadius: 4,
    shadowOffset: { width: 0, height: 2 },
    elevation: 1,
  },
  bubbleUser: {
    backgroundColor: COLORS.primary,
    borderBottomRightRadius: 4,
  },
  bubbleAI: {
    backgroundColor: '#fff',
    borderBottomLeftRadius: 4,
    borderWidth: 1,
    borderColor: '#E8F5E9',
  },
  bubbleVoice: {
    borderWidth: 1.5,
    borderColor: COLORS.primaryLight,
  },
  bubbleText: { fontSize: 15, lineHeight: 22 },
  bubbleTextUser: { color: '#fff' },
  bubbleTextAI: { color: COLORS.textPrimary },
  voiceTag: { fontSize: 10, color: COLORS.primaryLight, marginBottom: 4, fontWeight: '600' },
  timestamp: { fontSize: 10, marginTop: 4 },
  timestampUser: { color: 'rgba(255,255,255,0.6)', textAlign: 'right' },
  timestampAI: { color: COLORS.textMuted },

  // Typing
  typingBubble: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 5,
    paddingVertical: 14,
    paddingHorizontal: 16,
  },
  typingDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: COLORS.primaryLight,
  },

  // Suggestions
  suggestionsContainer: {
    paddingHorizontal: 16,
    paddingBottom: 8,
  },
  suggestionsLabel: {
    fontSize: 11,
    fontWeight: '700',
    color: COLORS.textMuted,
    textTransform: 'uppercase',
    letterSpacing: 0.8,
    marginBottom: 8,
  },
  suggestionChips: { flexDirection: 'row', flexWrap: 'wrap', gap: 8 },
  chip: {
    backgroundColor: '#E8F5E9',
    borderRadius: 20,
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderWidth: 1,
    borderColor: '#C8E6C9',
  },
  chipText: { fontSize: 13, color: COLORS.primaryDark, fontWeight: '600' },

  // Input bar
  inputBar: {
    backgroundColor: '#fff',
    borderTopWidth: 1,
    borderTopColor: '#E8EDE5',
    paddingTop: 8,
    paddingBottom: Platform.OS === 'ios' ? 4 : 8,
    paddingHorizontal: 12,
  },
  recordingBanner: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    backgroundColor: '#FFEBEE',
    borderRadius: 8,
    paddingVertical: 6,
    marginBottom: 8,
  },
  recordingDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: '#E53935',
  },
  recordingText: { fontSize: 13, color: '#C62828', fontWeight: '600' },
  inputRow: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    gap: 8,
    minHeight: 48,
  },
  textInput: {
    flex: 1,
    backgroundColor: '#F5F7F0',
    borderRadius: 24,
    paddingHorizontal: 16,
    paddingVertical: 10,
    fontSize: 15,
    color: COLORS.textPrimary,
    maxHeight: 100,
    borderWidth: 1.5,
    borderColor: '#DDE8D8',
  },
  sendBtn: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: COLORS.primary,
    alignItems: 'center',
    justifyContent: 'center',
    shadowColor: COLORS.primary,
    shadowOpacity: 0.35,
    shadowRadius: 6,
    shadowOffset: { width: 0, height: 3 },
    elevation: 4,
  },
  sendBtnIcon: { fontSize: 16, color: '#fff', marginLeft: 2 },
  disclaimer: {
    fontSize: 10,
    color: COLORS.textMuted,
    textAlign: 'center',
    marginTop: 6,
  },

  // Voice button
  voiceBtnWrapper: {
    width: 44,
    height: 44,
    alignItems: 'center',
    justifyContent: 'center',
  },
  voiceRing: {
    position: 'absolute',
    width: 44,
    height: 44,
    borderRadius: 22,
    borderWidth: 2,
    borderColor: '#E53935',
  },
  voiceBtn: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: COLORS.primary,
    alignItems: 'center',
    justifyContent: 'center',
    shadowColor: COLORS.primary,
    shadowOpacity: 0.35,
    shadowRadius: 6,
    shadowOffset: { width: 0, height: 3 },
    elevation: 4,
  },
  voiceBtnActive: { backgroundColor: '#E53935' },
  voiceBtnProcessing: { backgroundColor: COLORS.textMuted },
  voiceBtnIcon: { fontSize: 20 },
});

export default ChatScreen;
