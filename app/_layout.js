import { useEffect } from "react";
import { useRouter, useSegments } from "expo-router";
import { Drawer } from "expo-router/drawer";
import { AuthProvider, useAuth } from "../context/AuthContext";
import DrawerContent from "../components/Sidebar/DrawerContent";

// ── Onboarding Guard ─────────────────────────────────────────────────────────
// If the user hasn't set a name yet, send them to the profile setup screen.
function OnboardingGuard() {
  const { isProfileSet, isLoading } = useAuth();
  const segments = useSegments();
  const router = useRouter();

  useEffect(() => {
    if (isLoading) return;

    const inSetup = segments[0] === "(setup)";

    if (!isProfileSet && !inSetup) {
      router.replace("/(setup)/setup");
    } else if (isProfileSet && inSetup) {
      router.replace("/(main)/(tabs)/");
    }
  }, [isProfileSet, isLoading, segments]);

  return null;
}

// ── Root Layout ───────────────────────────────────────────────────────────────
export default function RootLayout() {
  return (
    <AuthProvider>
      <OnboardingGuard />
      <Drawer
        drawerContent={(props) => <DrawerContent {...props} />}
        screenOptions={{
          headerShown: false,
          drawerType: "slide",
          drawerStyle: { width: "78%" },
          overlayColor: "rgba(0,0,0,0.5)",
          swipeEdgeWidth: 80,
        }}
      >
        <Drawer.Screen
          name="(main)"
          options={{ drawerItemStyle: { display: "none" } }}
        />
        <Drawer.Screen
          name="(setup)"
          options={{
            drawerItemStyle: { display: "none" },
            swipeEnabled: false,
          }}
        />
      </Drawer>
    </AuthProvider>
  );
}
