import React, {
  createContext,
  useContext,
  useReducer,
  useEffect,
  useCallback,
} from "react";
import AsyncStorage from "@react-native-async-storage/async-storage";

// ── Constants ────────────────────────────────────────────────────────────────
const PROFILE_KEY = "farmer_profile";

const DEFAULT_PROFILE = {
  fullName: "",
  mobile: "",
  address: "",
  district: "",
  state: "",
  dateOfBirth: "",
  preferredLanguage: "en",
  profilePicture: null,
};

// ── State & Reducer ──────────────────────────────────────────────────────────
const initialState = {
  profile: null, // null = not yet loaded from storage
  isProfileSet: false, // true once user has filled in their details
  isLoading: true,
};

const reducer = (state, action) => {
  switch (action.type) {
    case "SET_LOADING":
      return { ...state, isLoading: action.payload };

    case "LOAD_PROFILE":
      return {
        ...state,
        profile: action.payload,
        isProfileSet: !!action.payload?.fullName,
        isLoading: false,
      };

    case "UPDATE_PROFILE":
      return {
        ...state,
        profile: { ...state.profile, ...action.payload },
        isProfileSet: true,
      };

    case "CLEAR_PROFILE":
      return {
        ...initialState,
        isLoading: false,
        profile: DEFAULT_PROFILE,
        isProfileSet: false,
      };

    default:
      return state;
  }
};

// ── Context ──────────────────────────────────────────────────────────────────
const ProfileContext = createContext(null);

export const AuthProvider = ({ children }) => {
  const [state, dispatch] = useReducer(reducer, initialState);

  // Load profile from AsyncStorage on mount
  useEffect(() => {
    loadProfile();
  }, []);

  const loadProfile = async () => {
    try {
      const raw = await AsyncStorage.getItem(PROFILE_KEY);
      const profile = raw ? JSON.parse(raw) : { ...DEFAULT_PROFILE };
      dispatch({ type: "LOAD_PROFILE", payload: profile });
    } catch {
      dispatch({ type: "LOAD_PROFILE", payload: { ...DEFAULT_PROFILE } });
    }
  };

  /**
   * Save / update profile fields and persist to AsyncStorage
   */
  const updateProfile = useCallback(
    async (fields) => {
      try {
        const current = state.profile ?? { ...DEFAULT_PROFILE };
        const updated = { ...current, ...fields };
        await AsyncStorage.setItem(PROFILE_KEY, JSON.stringify(updated));
        dispatch({ type: "UPDATE_PROFILE", payload: fields });
        return updated;
      } catch (err) {
        throw new Error("Failed to save profile. Please try again.");
      }
    },
    [state.profile],
  );

  /**
   * Wipe local profile (acts as "logout" / reset)
   */
  const clearProfile = useCallback(async () => {
    await AsyncStorage.removeItem(PROFILE_KEY);
    dispatch({ type: "CLEAR_PROFILE" });
  }, []);

  return (
    <ProfileContext.Provider
      value={{
        ...state,
        // Alias `profile` as `user` so existing components need minimal changes
        user: state.profile,
        updateProfile,
        updateUser: updateProfile, // backward-compat alias
        clearProfile,
        logout: clearProfile, // backward-compat alias
      }}
    >
      {children}
    </ProfileContext.Provider>
  );
};

export const useAuth = () => {
  const ctx = useContext(ProfileContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
};
