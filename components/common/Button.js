import React from "react";
import {
  TouchableOpacity,
  Text,
  ActivityIndicator,
  StyleSheet,
  View,
} from "react-native";
import { COLORS, RADIUS, FONT } from "../../constants";

const VARIANTS = {
  primary: {
    bg: COLORS.primary,
    text: "#fff",
    border: COLORS.primary,
  },
  secondary: {
    bg: COLORS.secondary,
    text: "#fff",
    border: COLORS.secondary,
  },
  outline: {
    bg: "transparent",
    text: COLORS.primary,
    border: COLORS.primary,
  },
  ghost: {
    bg: "transparent",
    text: COLORS.primary,
    border: "transparent",
  },
  danger: {
    bg: COLORS.error,
    text: "#fff",
    border: COLORS.error,
  },
  surface: {
    bg: COLORS.surface,
    text: COLORS.textPrimary,
    border: COLORS.border,
  },
};

const SIZES = {
  small: { px: 16, py: 8, fontSize: 13, height: 38 },
  medium: { px: 20, py: 13, fontSize: 15, height: 48 },
  large: { px: 24, py: 15, fontSize: 16, height: 54 },
};

const Button = ({
  title,
  onPress,
  loading = false,
  disabled = false,
  variant = "primary",
  size = "medium",
  style,
  textStyle,
  leftIcon,
  fullWidth = true,
}) => {
  const v = VARIANTS[variant] ?? VARIANTS.primary;
  const s = SIZES[size] ?? SIZES.medium;
  const isDisabled = disabled || loading;

  return (
    <TouchableOpacity
      onPress={onPress}
      disabled={isDisabled}
      activeOpacity={0.78}
      style={[
        styles.base,
        {
          backgroundColor: v.bg,
          borderColor: v.border,
          paddingHorizontal: s.px,
          paddingVertical: s.py,
          height: s.height,
          alignSelf: fullWidth ? "stretch" : "auto",
        },
        isDisabled && styles.disabled,
        style,
      ]}
    >
      {loading ? (
        <ActivityIndicator color={v.text} size="small" />
      ) : (
        <View style={styles.inner}>
          {leftIcon ? <View style={styles.leftIcon}>{leftIcon}</View> : null}
          <Text
            style={[
              styles.text,
              { color: v.text, fontSize: s.fontSize },
              textStyle,
            ]}
          >
            {title}
          </Text>
        </View>
      )}
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  base: {
    borderRadius: RADIUS.md,
    borderWidth: 1.5,
    alignItems: "center",
    justifyContent: "center",
  },
  inner: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
  },
  leftIcon: {},
  text: {
    fontWeight: FONT.bold,
    letterSpacing: 0.2,
  },
  disabled: { opacity: 0.48 },
});

export default Button;
