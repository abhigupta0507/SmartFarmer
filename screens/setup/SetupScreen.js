import React, { useState } from "react";
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  Alert,
  KeyboardAvoidingView,
  Platform,
  SafeAreaView,
  StatusBar,
} from "react-native";
import { useAuth } from "../../context/AuthContext";
import Input from "../../components/common/Input";
import Button from "../../components/common/Button";
import { COLORS, SHADOW, FONT, RADIUS } from "../../constants";

const FIELDS = [
  {
    key: "fullName",
    label: "Full Name",
    placeholder: "Enter your full name",
    caps: "words",
    kb: "default",
    required: true,
  },
  {
    key: "mobile",
    label: "Mobile Number",
    placeholder: "+919876543210",
    caps: "none",
    kb: "phone-pad",
    required: false,
  },
  {
    key: "address",
    label: "Address",
    placeholder: "Village / Town / City",
    caps: "sentences",
    kb: "default",
    required: true,
    multi: true,
  },
  {
    key: "district",
    label: "District",
    placeholder: "Enter district",
    caps: "words",
    kb: "default",
    required: true,
  },
  {
    key: "state",
    label: "State",
    placeholder: "Enter state",
    caps: "words",
    kb: "default",
    required: true,
  },
  {
    key: "dateOfBirth",
    label: "Date of Birth",
    placeholder: "YYYY-MM-DD",
    caps: "none",
    kb: "numeric",
    required: false,
  },
];

const SetupScreen = () => {
  const { updateProfile } = useAuth();
  const [loading, setLoading] = useState(false);
  const [errors, setErrors] = useState({});
  const [form, setForm] = useState({
    fullName: "",
    mobile: "",
    address: "",
    district: "",
    state: "",
    dateOfBirth: "",
  });

  const updateField = (key) => (val) => {
    setForm((p) => ({ ...p, [key]: val }));
    if (errors[key]) setErrors((p) => ({ ...p, [key]: null }));
  };

  const validate = () => {
    const e = {};
    if (!form.fullName.trim()) e.fullName = "Full name is required";
    if (!form.address.trim()) e.address = "Address is required";
    if (!form.district.trim()) e.district = "District is required";
    if (!form.state.trim()) e.state = "State is required";
    if (form.mobile.trim() && !/^\+?[1-9]\d{9,14}$/.test(form.mobile.trim()))
      e.mobile = "Format: +919876543210";
    if (
      form.dateOfBirth.trim() &&
      !/^\d{4}-\d{2}-\d{2}$/.test(form.dateOfBirth.trim())
    )
      e.dateOfBirth = "Format: YYYY-MM-DD";
    setErrors(e);
    return Object.keys(e).length === 0;
  };

  const handleSave = async () => {
    if (!validate()) return;
    setLoading(true);
    try {
      await updateProfile({
        fullName: form.fullName.trim(),
        mobile: form.mobile.trim(),
        address: form.address.trim(),
        district: form.district.trim(),
        state: form.state.trim(),
        dateOfBirth: form.dateOfBirth.trim(),
      });
    } catch (err) {
      Alert.alert("Error", err.message || "Could not save. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <SafeAreaView style={styles.safe}>
      <StatusBar barStyle="dark-content" backgroundColor={COLORS.background} />
      <KeyboardAvoidingView
        behavior={Platform.OS === "ios" ? "padding" : undefined}
        style={{ flex: 1 }}
      >
        <ScrollView
          contentContainerStyle={styles.content}
          showsVerticalScrollIndicator={false}
          keyboardShouldPersistTaps="handled"
        >
          {/* Header */}
          <View style={styles.header}>
            <View style={styles.logoCircle}>
              <Text style={styles.logoEmoji}>🌾</Text>
            </View>
            <Text style={styles.title}>Welcome to SmartFarmer</Text>
            <Text style={styles.subtitle}>
              Fill in your details to get started.{"\n"}You can update these
              anytime from your profile.
            </Text>
          </View>

          {/* Form card */}
          <View style={styles.card}>
            {FIELDS.map(
              ({ key, label, placeholder, caps, kb, required, multi }) => (
                <Input
                  key={key}
                  label={`${label}${required ? " *" : ""}`}
                  value={form[key]}
                  onChangeText={updateField(key)}
                  placeholder={placeholder}
                  autoCapitalize={caps}
                  keyboardType={kb}
                  error={errors[key]}
                  multiline={multi}
                  numberOfLines={multi ? 2 : undefined}
                />
              ),
            )}

            <Button
              title="Get Started →"
              onPress={handleSave}
              loading={loading}
              style={styles.btn}
              size="large"
            />
            <Text style={styles.hint}>* Required fields</Text>
          </View>

          <View style={{ height: 40 }} />
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: COLORS.background },
  content: { padding: 20, paddingTop: 16 },

  header: { alignItems: "center", marginBottom: 28 },
  logoCircle: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: COLORS.primaryFaded,
    alignItems: "center",
    justifyContent: "center",
    marginBottom: 16,
  },
  logoEmoji: { fontSize: 40 },
  title: {
    fontSize: 24,
    fontWeight: "800",
    color: COLORS.textPrimary,
    marginBottom: 8,
    textAlign: "center",
  },
  subtitle: {
    fontSize: 14,
    color: COLORS.textSecondary,
    textAlign: "center",
    lineHeight: 21,
  },

  card: {
    backgroundColor: COLORS.surface,
    borderRadius: 20,
    padding: 20,
    ...SHADOW.md,
  },
  btn: { marginTop: 6 },
  hint: {
    marginTop: 12,
    fontSize: 12,
    color: COLORS.textMuted,
    textAlign: "center",
  },
});

export default SetupScreen;
