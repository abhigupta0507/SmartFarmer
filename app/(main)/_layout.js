import { Tabs } from "expo-router";
import { Text, StyleSheet } from "react-native";
import { COLORS } from "../../constants";

const TabIcon = ({ emoji, label, focused }) => (
  <>
    <Text style={[styles.emoji, focused && styles.emojiFocused]}>{emoji}</Text>
    <Text style={[styles.label, focused && styles.labelFocused]}>{label}</Text>
  </>
);

export default function TabsLayout() {
  return (
    <Tabs
      screenOptions={{
        headerShown: false,
        tabBarShowLabel: false,
        tabBarActiveTintColor: COLORS.primary,
        tabBarInactiveTintColor: COLORS.textMuted,
        tabBarStyle: styles.tabBar,
      }}
    >
      <Tabs.Screen
        name="index"
        options={{
          tabBarIcon: ({ focused }) => (
            <TabIcon emoji="🏠" label="Home" focused={focused} />
          ),
        }}
      />
      <Tabs.Screen
        name="weather"
        options={{
          tabBarIcon: ({ focused }) => (
            <TabIcon emoji="🌤️" label="Weather" focused={focused} />
          ),
        }}
      />
      <Tabs.Screen
        name="market"
        options={{
          tabBarIcon: ({ focused }) => (
            <TabIcon emoji="🛒" label="Market" focused={focused} />
          ),
        }}
      />
      <Tabs.Screen
        name="profile"
        options={{
          tabBarIcon: ({ focused }) => (
            <TabIcon emoji="👤" label="Profile" focused={focused} />
          ),
        }}
      />

      {/* Hide advisory from tabs (accessible via drawer/dashboard) */}
      <Tabs.Screen name="advisory" options={{ href: null }} />
    </Tabs>
  );
}

const styles = StyleSheet.create({
  tabBar: {
    height: 68,
    backgroundColor: COLORS.surface,
    borderTopWidth: 1,
    borderTopColor: COLORS.border,
    paddingTop: 6,
    paddingBottom: 10,
    elevation: 12,
    shadowColor: "#000",
    shadowOpacity: 0.08,
    shadowRadius: 12,
    shadowOffset: { width: 0, height: -3 },
  },
  emoji: {
    fontSize: 22,
    opacity: 0.45,
    marginBottom: 2,
  },
  emojiFocused: {
    opacity: 1,
  },
  label: {
    fontSize: 10,
    fontWeight: "600",
    color: COLORS.textMuted,
    letterSpacing: 0.3,
  },
  labelFocused: {
    color: COLORS.primary,
  },
});
