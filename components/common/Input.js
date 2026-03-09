import React, { useState } from "react";
import {
  View,
  TextInput,
  Text,
  TouchableOpacity,
  StyleSheet,
} from "react-native";
import { COLORS, RADIUS, FONT } from "../../constants";

const Input = ({
  label,
  value,
  onChangeText,
  placeholder,
  secureTextEntry,
  error,
  keyboardType = "default",
  autoCapitalize = "sentences",
  editable = true,
  leftIcon,
  rightIcon,
  onRightIconPress,
  multiline,
  numberOfLines,
  style,
  ...props
}) => {
  const [isFocused, setIsFocused] = useState(false);

  return (
    <View style={[styles.wrapper, style]}>
      {label ? <Text style={styles.label}>{label}</Text> : null}
      <View
        style={[
          styles.container,
          isFocused && styles.focused,
          error && styles.errorBorder,
          !editable && styles.disabled,
        ]}
      >
        {leftIcon ? <View style={styles.leftIcon}>{leftIcon}</View> : null}
        <TextInput
          value={value}
          onChangeText={onChangeText}
          placeholder={placeholder}
          placeholderTextColor={COLORS.textMuted}
          secureTextEntry={secureTextEntry}
          keyboardType={keyboardType}
          autoCapitalize={autoCapitalize}
          editable={editable}
          multiline={multiline}
          numberOfLines={numberOfLines}
          onFocus={() => setIsFocused(true)}
          onBlur={() => setIsFocused(false)}
          style={[
            styles.input,
            !editable && styles.disabledText,
            multiline && styles.multiline,
          ]}
          {...props}
        />
        {rightIcon ? (
          <TouchableOpacity
            onPress={onRightIconPress}
            style={styles.rightIcon}
            activeOpacity={0.7}
          >
            {rightIcon}
          </TouchableOpacity>
        ) : null}
      </View>
      {error ? <Text style={styles.errorText}>{error}</Text> : null}
    </View>
  );
};

const styles = StyleSheet.create({
  wrapper: { marginBottom: 14 },

  label: {
    fontSize: 13,
    fontWeight: FONT.semibold,
    color: COLORS.textSecondary,
    marginBottom: 6,
    letterSpacing: 0.2,
  },

  container: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: COLORS.inputBg,
    borderRadius: RADIUS.md,
    borderWidth: 1.5,
    borderColor: COLORS.border,
    paddingHorizontal: 14,
    minHeight: 50,
  },
  focused: { borderColor: COLORS.primary, backgroundColor: COLORS.surface },
  errorBorder: { borderColor: COLORS.error },
  disabled: { backgroundColor: "#ECECEC", opacity: 0.75 },

  input: {
    flex: 1,
    fontSize: 15,
    fontWeight: FONT.regular,
    color: COLORS.textPrimary,
    paddingVertical: 12,
  },
  disabledText: { color: COLORS.textMuted },
  multiline: { textAlignVertical: "top", paddingTop: 12 },

  leftIcon: { marginRight: 10 },
  rightIcon: { marginLeft: 10, padding: 2 },

  errorText: {
    fontSize: 12,
    fontWeight: FONT.medium,
    color: COLORS.error,
    marginTop: 5,
    marginLeft: 2,
  },
});

export default Input;
