import React from "react";
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  SafeAreaView,
  StatusBar,
} from "react-native";
import { useRouter } from "expo-router";
import { useAuth } from "../../context/AuthContext";
import { COLORS, SHADOW, FONT, RADIUS } from "../../constants";

const getGreeting = () => {
  const h = new Date().getHours();
  if (h < 12) return "Good Morning,";
  if (h < 17) return "Good Afternoon,";
  return "Good Evening,";
};

const StatCard = ({ emoji, value, label, bg }) => (
  <View style={[styles.statCard, { backgroundColor: bg }]}>
    <Text style={styles.statEmoji}>{emoji}</Text>
    <Text style={styles.statValue}>{value}</Text>
    <Text style={styles.statLabel}>{label}</Text>
  </View>
);

const ActionCard = ({ emoji, title, subtitle, accentColor, onPress }) => (
  <TouchableOpacity
    style={[styles.actionCard, { borderLeftColor: accentColor }]}
    onPress={onPress}
    activeOpacity={0.82}
  >
    <View
      style={[styles.actionIconBox, { backgroundColor: accentColor + "1A" }]}
    >
      <Text style={styles.actionEmoji}>{emoji}</Text>
    </View>
    <View style={styles.actionText}>
      <Text style={styles.actionTitle}>{title}</Text>
      <Text style={styles.actionSub}>{subtitle}</Text>
    </View>
    <Text style={styles.actionArrow}>›</Text>
  </TouchableOpacity>
);

const DashboardScreen = () => {
  const router = useRouter();
  const { user } = useAuth();
  const firstName = user?.fullName?.trim().split(" ")[0] || "Farmer";
  const locationLine = [user?.district, user?.state].filter(Boolean).join(", ");

  const actions = [
    {
      emoji: "🔍",
      title: "Pest Detection",
      subtitle: "Scan crop for disease & pests",
      color: COLORS.error,
      route: "/(main)/(tabs)/pest-detection",
    },
    {
      emoji: "🌤️",
      title: "Weather Forecast",
      subtitle: "7-day hyperlocal forecast",
      color: "#0288D1",
      route: "/(main)/(tabs)/weather",
    },
    {
      emoji: "📊",
      title: "Crop Advisory",
      subtitle: "Expert recommendations",
      color: COLORS.primary,
      route: "/(main)/(tabs)/advisory",
    },
    {
      emoji: "🛒",
      title: "Market Prices",
      subtitle: "Today's mandi rates",
      color: COLORS.secondary,
      route: "/(main)/(tabs)/market",
    },
  ];

  return (
    <SafeAreaView style={styles.safe}>
      <StatusBar barStyle="dark-content" backgroundColor={COLORS.background} />
      <ScrollView
        style={styles.scroll}
        contentContainerStyle={styles.content}
        showsVerticalScrollIndicator={false}
      >
        {/* Header */}
        <View style={styles.header}>
          <View style={styles.headerLeft}>
            <Text style={styles.greeting}>{getGreeting()}</Text>
            <Text style={styles.name}>{firstName} 👋</Text>
            {locationLine ? (
              <View style={styles.locRow}>
                <Text style={styles.locPin}>📍</Text>
                <Text style={styles.locText}>{locationLine}</Text>
              </View>
            ) : null}
          </View>
          <TouchableOpacity
            onPress={() => router.push("/(main)")}
            style={styles.menuBtn}
            activeOpacity={0.7}
          >
            <View style={styles.menuBar} />
            <View style={[styles.menuBar, { width: 16 }]} />
            <View style={styles.menuBar} />
          </TouchableOpacity>
        </View>

        {/* Stats */}
        <View style={styles.statsRow}>
          <StatCard
            emoji="🌱"
            value="3"
            label="Crops"
            bg={COLORS.primaryFaded}
          />
          <StatCard
            emoji="⚠️"
            value="0"
            label="Alerts"
            bg={COLORS.secondaryFaded}
          />
          <StatCard emoji="💧" value="72%" label="Humidity" bg="#E3F2FD" />
        </View>

        {/* Actions */}
        <Text style={styles.sectionTitle}>Quick Actions</Text>
        {actions.map((a) => (
          <ActionCard
            key={a.title}
            emoji={a.emoji}
            title={a.title}
            subtitle={a.subtitle}
            accentColor={a.color}
            onPress={() => router.push(a.route)}
          />
        ))}

        <View style={{ height: 24 }} />
      </ScrollView>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: COLORS.background },
  scroll: { flex: 1 },
  content: { padding: 20, paddingTop: 12 },

  header: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "flex-start",
    marginBottom: 24,
  },
  headerLeft: { flex: 1, paddingRight: 12 },
  greeting: {
    fontSize: 13,
    fontWeight: "500",
    color: COLORS.textMuted,
    marginBottom: 2,
  },
  name: {
    fontSize: 26,
    fontWeight: "800",
    color: COLORS.textPrimary,
    marginBottom: 6,
  },
  locRow: { flexDirection: "row", alignItems: "center", gap: 4 },
  locPin: { fontSize: 12 },
  locText: { fontSize: 12, fontWeight: "600", color: COLORS.textSecondary },

  menuBtn: {
    width: 44,
    height: 44,
    backgroundColor: COLORS.surface,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: COLORS.border,
    alignItems: "center",
    justifyContent: "center",
    gap: 5,
    ...SHADOW.sm,
  },
  menuBar: {
    width: 20,
    height: 2,
    backgroundColor: COLORS.textPrimary,
    borderRadius: 2,
  },

  statsRow: { flexDirection: "row", gap: 10, marginBottom: 28 },
  statCard: {
    flex: 1,
    borderRadius: 14,
    padding: 14,
    alignItems: "center",
    ...SHADOW.sm,
  },
  statEmoji: { fontSize: 20, marginBottom: 6 },
  statValue: {
    fontSize: 18,
    fontWeight: "800",
    color: COLORS.textPrimary,
    marginBottom: 2,
  },
  statLabel: {
    fontSize: 10,
    fontWeight: "600",
    color: COLORS.textMuted,
    textAlign: "center",
  },

  sectionTitle: {
    fontSize: 16,
    fontWeight: "700",
    color: COLORS.textPrimary,
    marginBottom: 12,
    letterSpacing: 0.2,
  },

  actionCard: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: COLORS.surface,
    borderRadius: 14,
    padding: 15,
    marginBottom: 10,
    borderLeftWidth: 4,
    gap: 14,
    ...SHADOW.sm,
  },
  actionIconBox: {
    width: 48,
    height: 48,
    borderRadius: 12,
    alignItems: "center",
    justifyContent: "center",
  },
  actionEmoji: { fontSize: 22 },
  actionText: { flex: 1 },
  actionTitle: {
    fontSize: 15,
    fontWeight: "700",
    color: COLORS.textPrimary,
    marginBottom: 2,
  },
  actionSub: { fontSize: 12, color: COLORS.textSecondary },
  actionArrow: { fontSize: 24, color: COLORS.textMuted },
});

export default DashboardScreen;
