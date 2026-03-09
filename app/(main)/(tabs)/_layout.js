import { Tabs } from "expo-router";
import { View, Text, StyleSheet } from "react-native";
import { COLORS } from "../../../constants";

const TabIcon = ({ emoji, label, focused }) => (
  <View style={styles.tabItem}>
    <Text style={[styles.emoji, focused && styles.emojiFocused]}>{emoji}</Text>
    <Text style={[styles.label, focused && styles.labelFocused]}>{label}</Text>
  </View>
);

export default function TabsLayout() {
  return (
    <Tabs
      screenOptions={{
        headerShown: false,
        // Use the built-in label but hide it — we render our own inside tabBarIcon
        tabBarShowLabel: false,
        tabBarStyle: styles.tabBar,
        // Make each tab button fill equal width
        tabBarItemStyle: styles.tabBarItem,
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

      {/* Keep advisory registered but hidden from tab bar */}
      <Tabs.Screen name="advisory" options={{ href: null }} />
    </Tabs>
  );
}

const styles = StyleSheet.create({
  tabBar: {
    height: 64,
    backgroundColor: COLORS.surface,
    borderTopWidth: 1,
    borderTopColor: COLORS.border,
    elevation: 3,
    shadowColor: "#000",
    shadowOpacity: 0.08,
    shadowRadius: 12,
    shadowOffset: { width: 0, height: -3 },
    paddingTop: 0,
    paddingBottom: 0,
  },

  // Each tab button fills its flex cell fully
  tabBarItem: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    paddingTop: 0,
    paddingBottom: 0,
    height: 64,
  },

  // The inner View inside each tab icon
  tabItem: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    gap: 3,
    paddingVertical: 8,
    width: 72,
  },

  emoji: {
    fontSize: 22,
    opacity: 0.4,
  },
  emojiFocused: {
    opacity: 1,
  },
  label: {
    fontSize: 10,
    fontWeight: "600",
    color: COLORS.textMuted,
    letterSpacing: 0.2,
  },
  labelFocused: {
    color: COLORS.primary,
  },
});
