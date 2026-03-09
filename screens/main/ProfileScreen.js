import React, { useState } from "react";
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  Alert,
  TouchableOpacity,
  SafeAreaView,
  StatusBar,
} from "react-native";
import { useAuth } from "../../context/AuthContext";
import Input from "../../components/common/Input";
import Button from "../../components/common/Button";
import { COLORS, SHADOW, FONT, RADIUS } from "../../constants";

const SectionLabel = ({ title }) => (
  <Text style={styles.sectionLabel}>{title}</Text>
);

const InfoRow = ({ label, value }) => (
  <View style={styles.infoRow}>
    <Text style={styles.infoLabel}>{label}</Text>
    <Text style={styles.infoValue}>{value || "—"}</Text>
  </View>
);

const ProfileScreen = () => {
  const { user, updateProfile, clearProfile } = useAuth();
  const [isEditing, setIsEditing] = useState(false);
  const [loading, setLoading] = useState(false);
  const [errors, setErrors] = useState({});

  const buildForm = (src) => ({
    fullName: src?.fullName ?? "",
    mobile: src?.mobile ?? "",
    address: src?.address ?? "",
    district: src?.district ?? "",
    state: src?.state ?? "",
    dateOfBirth: src?.dateOfBirth ?? "",
  });

  const [form, setForm] = useState(() => buildForm(user));

  const updateField = (field) => (value) => {
    setForm((p) => ({ ...p, [field]: value }));
    if (errors[field]) setErrors((p) => ({ ...p, [field]: null }));
  };

  const validate = () => {
    const e = {};
    if (!form.fullName.trim()) e.fullName = "Full name is required";
    if (!form.address.trim()) e.address = "Address is required";
    if (!form.district.trim()) e.district = "District is required";
    if (!form.state.trim()) e.state = "State is required";
    if (form.mobile.trim() && !/^\+?[1-9]\d{9,14}$/.test(form.mobile.trim()))
      e.mobile = "Use format: +919876543210";
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
      setIsEditing(false);
      Alert.alert("Saved ✓", "Your profile has been updated.");
    } catch (err) {
      Alert.alert("Error", err.message || "Failed to save.");
    } finally {
      setLoading(false);
    }
  };

  const handleCancel = () => {
    setForm(buildForm(user));
    setErrors({});
    setIsEditing(false);
  };

  const handleReset = () => {
    Alert.alert(
      "Reset Profile",
      "This will clear all saved data and return you to the setup screen.",
      [
        { text: "Cancel", style: "cancel" },
        { text: "Reset", style: "destructive", onPress: clearProfile },
      ],
    );
  };

  const initials =
    user?.fullName
      ?.trim()
      .split(" ")
      .map((w) => w[0])
      .slice(0, 2)
      .join("")
      .toUpperCase() || "?";

  return (
    <SafeAreaView style={styles.safe}>
      <StatusBar barStyle="dark-content" backgroundColor={COLORS.background} />
      <ScrollView
        style={styles.scroll}
        contentContainerStyle={styles.content}
        showsVerticalScrollIndicator={false}
      >
        {/* Avatar card */}
        <View style={styles.avatarCard}>
          <View style={styles.avatar}>
            <Text style={styles.avatarText}>{initials}</Text>
          </View>
          <Text style={styles.userName}>{user?.fullName || "Farmer"}</Text>
          {user?.district || user?.state ? (
            <Text style={styles.userLocation}>
              📍 {[user.district, user.state].filter(Boolean).join(", ")}
            </Text>
          ) : null}
          <View style={styles.verifiedBadge}>
            <Text style={styles.verifiedText}>🌾 SmartFarmer</Text>
          </View>
        </View>

        {/* Edit controls */}
        {isEditing ? (
          <View style={styles.editRow}>
            <Button
              title="Save Changes"
              onPress={handleSave}
              loading={loading}
              style={{ flex: 1 }}
              size="medium"
            />
            <Button
              title="Cancel"
              onPress={handleCancel}
              variant="outline"
              style={{ flex: 1 }}
              size="medium"
            />
          </View>
        ) : (
          <Button
            title="Edit Profile"
            onPress={() => setIsEditing(true)}
            variant="outline"
            style={styles.editBtn}
            size="medium"
          />
        )}

        {/* View mode — clean info rows */}
        {!isEditing ? (
          <>
            <SectionLabel title="Personal Info" />
            <View style={styles.infoCard}>
              <InfoRow label="Full Name" value={user?.fullName} />
              <View style={styles.divider} />
              <InfoRow label="Mobile" value={user?.mobile} />
              <View style={styles.divider} />
              <InfoRow label="Date of Birth" value={user?.dateOfBirth} />
            </View>

            <SectionLabel title="Location" />
            <View style={styles.infoCard}>
              <InfoRow label="Address" value={user?.address} />
              <View style={styles.divider} />
              <InfoRow label="District" value={user?.district} />
              <View style={styles.divider} />
              <InfoRow label="State" value={user?.state} />
            </View>
          </>
        ) : (
          <>
            {/* Edit mode — input fields */}
            <SectionLabel title="Personal Info" />
            <Input
              label="Full Name *"
              value={form.fullName}
              onChangeText={updateField("fullName")}
              error={errors.fullName}
              autoCapitalize="words"
            />
            <Input
              label="Mobile"
              value={form.mobile}
              onChangeText={updateField("mobile")}
              error={errors.mobile}
              keyboardType="phone-pad"
              autoCapitalize="none"
              placeholder="+919876543210"
            />
            <Input
              label="Date of Birth"
              value={form.dateOfBirth}
              onChangeText={updateField("dateOfBirth")}
              error={errors.dateOfBirth}
              keyboardType="numeric"
              placeholder="YYYY-MM-DD"
            />

            <SectionLabel title="Location" />
            <Input
              label="Address *"
              value={form.address}
              onChangeText={updateField("address")}
              error={errors.address}
              multiline
              numberOfLines={2}
            />
            <Input
              label="District *"
              value={form.district}
              onChangeText={updateField("district")}
              error={errors.district}
              autoCapitalize="words"
            />
            <Input
              label="State *"
              value={form.state}
              onChangeText={updateField("state")}
              error={errors.state}
              autoCapitalize="words"
            />
          </>
        )}

        {/* Settings */}
        <SectionLabel title="Settings" />
        <View style={styles.settingCard}>
          <TouchableOpacity style={styles.settingRow} activeOpacity={0.7}>
            <Text style={styles.settingEmoji}>🌐</Text>
            <View style={{ flex: 1 }}>
              <Text style={styles.settingTitle}>Language</Text>
              <Text style={styles.settingValue}>English (Coming soon)</Text>
            </View>
            <Text style={styles.settingArrow}>›</Text>
          </TouchableOpacity>
        </View>

        {/* Danger */}
        <SectionLabel title="Danger Zone" />
        <Button
          title="Reset Profile & Data"
          onPress={handleReset}
          variant="danger"
          size="medium"
          style={styles.dangerBtn}
        />

        <View style={{ height: 40 }} />
      </ScrollView>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: COLORS.background },
  scroll: { flex: 1 },
  content: { padding: 20, paddingTop: 12 },

  avatarCard: {
    backgroundColor: COLORS.surface,
    borderRadius: 20,
    padding: 24,
    alignItems: "center",
    marginBottom: 16,
    ...SHADOW.sm,
  },
  avatar: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: COLORS.primary,
    alignItems: "center",
    justifyContent: "center",
    marginBottom: 12,
  },
  avatarText: { fontSize: 28, fontWeight: "800", color: "#fff" },
  userName: {
    fontSize: 20,
    fontWeight: "800",
    color: COLORS.textPrimary,
    marginBottom: 4,
  },
  userLocation: { fontSize: 13, color: COLORS.textSecondary, marginBottom: 10 },
  verifiedBadge: {
    backgroundColor: COLORS.primaryFaded,
    borderRadius: 20,
    paddingHorizontal: 14,
    paddingVertical: 5,
  },
  verifiedText: { fontSize: 12, fontWeight: "700", color: COLORS.primaryDark },

  editBtn: { marginBottom: 20 },
  editRow: { flexDirection: "row", gap: 10, marginBottom: 20 },

  sectionLabel: {
    fontSize: 12,
    fontWeight: "700",
    color: COLORS.textMuted,
    textTransform: "uppercase",
    letterSpacing: 1,
    marginBottom: 10,
    marginTop: 6,
  },

  infoCard: {
    backgroundColor: COLORS.surface,
    borderRadius: 14,
    paddingHorizontal: 16,
    marginBottom: 16,
    ...SHADOW.sm,
  },
  infoRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    paddingVertical: 14,
  },
  infoLabel: {
    fontSize: 13,
    fontWeight: "600",
    color: COLORS.textMuted,
    flex: 1,
  },
  infoValue: {
    fontSize: 14,
    fontWeight: "500",
    color: COLORS.textPrimary,
    flex: 2,
    textAlign: "right",
  },
  divider: { height: 1, backgroundColor: COLORS.border },

  settingCard: {
    backgroundColor: COLORS.surface,
    borderRadius: 14,
    marginBottom: 16,
    ...SHADOW.sm,
  },
  settingRow: {
    flexDirection: "row",
    alignItems: "center",
    padding: 16,
    gap: 12,
  },
  settingEmoji: { fontSize: 20 },
  settingTitle: { fontSize: 14, fontWeight: "600", color: COLORS.textPrimary },
  settingValue: { fontSize: 12, color: COLORS.textMuted, marginTop: 2 },
  settingArrow: { fontSize: 20, color: COLORS.textMuted },

  dangerBtn: { marginBottom: 8 },
});

export default ProfileScreen;
