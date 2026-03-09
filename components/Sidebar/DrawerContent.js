import React from "react";
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
  StatusBar,
} from "react-native";
import { useRouter } from "expo-router";
import { useAuth } from "../../context/AuthContext";
import { COLORS, FONT } from "../../constants";

const NavItem = ({ emoji, label, onPress, isActive }) => (
  <TouchableOpacity
    style={[styles.navItem, isActive && styles.navItemActive]}
    onPress={onPress}
    activeOpacity={0.7}
  >
    <Text style={styles.navEmoji}>{emoji}</Text>
    <Text style={[styles.navLabel, isActive && styles.navLabelActive]}>
      {label}
    </Text>
    {isActive && <View style={styles.activeDot} />}
  </TouchableOpacity>
);

const SectionLabel = ({ title }) => (
  <Text style={styles.sectionLabel}>{title}</Text>
);

const DrawerContent = () => {
  const router = useRouter();
  const { user } = useAuth();

  const navigate = (path) => router.push(path);

  const initials =
    user?.fullName
      ?.trim()
      .split(" ")
      .map((w) => w[0])
      .slice(0, 2)
      .join("")
      .toUpperCase() || "?";

  const locationLine = [user?.district, user?.state].filter(Boolean).join(", ");

  return (
    <ScrollView
      style={styles.scroll}
      contentContainerStyle={styles.container}
      showsVerticalScrollIndicator={false}
    >
      {/* Profile header */}
      <View style={styles.profileSection}>
        <StatusBar barStyle="light-content" />
        <View style={styles.avatar}>
          <Text style={styles.avatarText}>{initials}</Text>
        </View>
        <Text style={styles.userName} numberOfLines={1}>
          {user?.fullName || "Farmer"}
        </Text>
        {user?.mobile ? (
          <Text style={styles.userMobile}>{user.mobile}</Text>
        ) : null}
        {locationLine ? (
          <View style={styles.locationRow}>
            <Text style={styles.locationPin}>📍</Text>
            <Text style={styles.locationText}>{locationLine}</Text>
          </View>
        ) : null}
      </View>

      {/* Nav */}
      <View style={styles.nav}>
        <SectionLabel title="Main" />
        <NavItem
          emoji="🏠"
          label="Dashboard"
          onPress={() => navigate("/(main)/(tabs)/")}
        />
        <NavItem
          emoji="🌤️"
          label="Weather"
          onPress={() => navigate("/(main)/(tabs)/weather")}
        />
        <NavItem
          emoji="🛒"
          label="Market Prices"
          onPress={() => navigate("/(main)/(tabs)/market")}
        />
        <NavItem
          emoji="📊"
          label="Crop Advisory"
          onPress={() => navigate("/(main)/(tabs)/advisory")}
        />

        <View style={styles.divider} />

        <SectionLabel title="Account" />
        <NavItem
          emoji="👤"
          label="My Profile"
          onPress={() => navigate("/(main)/(tabs)/profile")}
        />
        <NavItem emoji="🌐" label="Language" onPress={() => {}} />
        <NavItem emoji="❓" label="Help & Support" onPress={() => {}} />
      </View>

      <Text style={styles.version}>SmartFarmer v1.0.0</Text>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  scroll: { flex: 1, backgroundColor: COLORS.surface },
  container: { flexGrow: 1, paddingBottom: 32 },

  profileSection: {
    backgroundColor: COLORS.primaryDark,
    paddingHorizontal: 20,
    paddingTop: 52,
    paddingBottom: 24,
  },
  avatar: {
    width: 64,
    height: 64,
    borderRadius: 32,
    backgroundColor: "rgba(255,255,255,0.22)",
    alignItems: "center",
    justifyContent: "center",
    marginBottom: 14,
    borderWidth: 2,
    borderColor: "rgba(255,255,255,0.3)",
  },
  avatarText: { fontSize: 24, fontWeight: "800", color: "#fff" },
  userName: { fontSize: 18, fontWeight: "800", color: "#fff", marginBottom: 3 },
  userMobile: { fontSize: 13, color: "rgba(255,255,255,0.7)", marginBottom: 6 },
  locationRow: { flexDirection: "row", alignItems: "center", gap: 4 },
  locationPin: { fontSize: 12 },
  locationText: {
    fontSize: 12,
    fontWeight: "600",
    color: "rgba(255,255,255,0.75)",
  },

  nav: { padding: 12 },

  sectionLabel: {
    fontSize: 10,
    fontWeight: "700",
    color: COLORS.textMuted,
    letterSpacing: 1.2,
    textTransform: "uppercase",
    marginLeft: 12,
    marginBottom: 4,
    marginTop: 8,
  },

  navItem: {
    flexDirection: "row",
    alignItems: "center",
    paddingVertical: 12,
    paddingHorizontal: 12,
    borderRadius: 10,
    marginBottom: 2,
    position: "relative",
  },
  navItemActive: {
    backgroundColor: COLORS.primaryFaded,
  },
  navEmoji: { fontSize: 19, width: 30, textAlign: "center", marginRight: 12 },
  navLabel: { fontSize: 15, fontWeight: "500", color: COLORS.textPrimary },
  navLabelActive: { fontWeight: "700", color: COLORS.primaryDark },
  activeDot: {
    position: "absolute",
    right: 12,
    top: "50%",
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: COLORS.primary,
    marginTop: -3,
  },

  divider: {
    height: 1,
    backgroundColor: COLORS.border,
    marginVertical: 8,
    marginHorizontal: 12,
  },

  version: {
    textAlign: "center",
    fontSize: 11,
    color: COLORS.textMuted,
    marginTop: 24,
  },
});

export default DrawerContent;
