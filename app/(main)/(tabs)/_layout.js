import { Tabs } from 'expo-router';
import { View, Text } from 'react-native';
import { COLORS } from '../../../constants';

const TabIcon = ({ emoji, focused }) => (
  <Text style={{ fontSize: 22, opacity: focused ? 1 : 0.5 }}>{emoji}</Text>
);

export default function TabsLayout() {
  return (
    <Tabs
      screenOptions={{
        headerShown: false,
        tabBarShowLabel: false,
        tabBarStyle: {
          height: 64,
          backgroundColor: COLORS.surface,
          borderTopColor: COLORS.border,
          borderTopWidth: 1,
          elevation: 10,
          shadowOpacity: 0.08,
        },
      }}
    >
      {/* Left tab 1 */}
      <Tabs.Screen
        name="weather"
        options={{
          tabBarIcon: ({ focused }) => <TabIcon emoji="🌤️" focused={focused} />,
        }}
      />

      {/* Left tab 2 */}
      <Tabs.Screen
        name="market"
        options={{
          tabBarIcon: ({ focused }) => <TabIcon emoji="🛒" focused={focused} />,
        }}
      />

      {/* Center — big Pest Detection button */}
      <Tabs.Screen
        name="index"
        options={{
          tabBarIcon: ({ focused }) => (
            <View style={{
              width: 60, height: 60, borderRadius: 30,
              backgroundColor: COLORS.primary,
              alignItems: 'center', justifyContent: 'center',
              bottom: 16,
              shadowColor: COLORS.primary,
              shadowOpacity: 0.4,
              shadowRadius: 8,
              shadowOffset: { width: 0, height: 4 },
              elevation: 8,
            }}>
              <Text style={{ fontSize: 26 }}>🔍</Text>
            </View>
          ),
        }}
      />

      {/* Right tab 1 */}
      <Tabs.Screen
        name="advisory"
        options={{
          tabBarIcon: ({ focused }) => <TabIcon emoji="📊" focused={focused} />,
        }}
      />

      {/* Right tab 2 */}
      <Tabs.Screen
        name="profile"
        options={{
          tabBarIcon: ({ focused }) => <TabIcon emoji="👤" focused={focused} />,
        }}
      />

      <Tabs.Screen
          name="edgeai"
          options={{
            tabBarIcon: ({ focused }) => <TabIcon emoji="🤖" focused={focused} />,
          }}
        />
    </Tabs>
  );
}
