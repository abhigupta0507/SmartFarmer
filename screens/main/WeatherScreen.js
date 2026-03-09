import React, { useState, useEffect, useCallback } from "react";
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  SafeAreaView,
  TouchableOpacity,
  ActivityIndicator,
  RefreshControl,
  StatusBar,
} from "react-native";
import { useAuth } from "../../context/AuthContext";
import apiService from "../../services/api.service";
import { COLORS, SHADOW, FONT, RADIUS } from "../../constants";

/* ─── Weather Icon Mapping ────────────────────────────────────────────────── */
const getWeatherEmoji = (condition = "") => {
  const c = condition.toLowerCase();
  if (c.includes("sunny") || c.includes("clear")) return "☀️";
  if (c.includes("partly cloudy")) return "⛅";
  if (c.includes("cloudy") || c.includes("overcast")) return "☁️";
  if (c.includes("rain") || c.includes("drizzle")) return "🌧️";
  if (c.includes("thunder") || c.includes("storm")) return "⛈️";
  if (c.includes("snow") || c.includes("sleet")) return "❄️";
  if (c.includes("fog") || c.includes("mist")) return "🌫️";
  if (c.includes("wind")) return "💨";
  return "🌡️";
};

const getUVLabel = (uv) => {
  if (uv <= 2) return { label: "Low", color: "#43A047" };
  if (uv <= 5) return { label: "Moderate", color: "#FB8C00" };
  if (uv <= 7) return { label: "High", color: "#E53935" };
  if (uv <= 10) return { label: "Very High", color: "#8E24AA" };
  return { label: "Extreme", color: "#6D1A36" };
};

/* ─── Sub-components ──────────────────────────────────────────────────────── */

const StatTile = ({ emoji, label, value, unit }) => (
  <View style={styles.statTile}>
    <Text style={styles.statTileEmoji}>{emoji}</Text>
    <Text style={styles.statTileValue}>
      {value}
      <Text style={styles.statTileUnit}>{unit}</Text>
    </Text>
    <Text style={styles.statTileLabel}>{label}</Text>
  </View>
);

const ForecastDay = ({ day, isToday }) => {
  const date = new Date(day.date);
  const dayName = isToday
    ? "Today"
    : date.toLocaleDateString("en-IN", { weekday: "short" });
  const dateStr = date.toLocaleDateString("en-IN", {
    day: "numeric",
    month: "short",
  });

  return (
    <View style={[styles.forecastCard, isToday && styles.forecastCardToday]}>
      <Text style={[styles.forecastDay, isToday && styles.forecastDayToday]}>
        {dayName}
      </Text>
      <Text style={styles.forecastDate}>{dateStr}</Text>
      <Text style={styles.forecastEmoji}>{getWeatherEmoji(day.condition)}</Text>
      <Text
        style={[
          styles.forecastCondition,
          isToday && styles.forecastConditionToday,
        ]}
        numberOfLines={2}
      >
        {day.condition}
      </Text>
      <View style={styles.forecastTemps}>
        <Text style={styles.forecastMax}>{Math.round(day.max_temp_c)}°</Text>
        <Text style={styles.forecastMin}>{Math.round(day.min_temp_c)}°</Text>
      </View>
      {day.total_precip_mm > 0 ? (
        <Text style={styles.forecastRain}>💧 {day.total_precip_mm}mm</Text>
      ) : null}
    </View>
  );
};

const AlertBanner = ({ alert }) => (
  <View style={styles.alertBanner}>
    <Text style={styles.alertIcon}>⚠️</Text>
    <View style={styles.alertText}>
      <Text style={styles.alertTitle}>{alert.headline || "Weather Alert"}</Text>
      {alert.desc ? (
        <Text style={styles.alertDesc} numberOfLines={2}>
          {alert.desc}
        </Text>
      ) : null}
    </View>
  </View>
);

/* ─── Main Screen ─────────────────────────────────────────────────────────── */

const WeatherScreen = () => {
  const { user } = useAuth();

  const [weather, setWeather] = useState(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState(null);

  // Use district+state as location, fall back to state
  console.log(user);
  const location =
    [user?.district, user?.state].filter(Boolean).join(", ") || "Delhi";

  const fetchWeather = useCallback(
    async (isRefresh = false) => {
      try {
        isRefresh ? setRefreshing(true) : setLoading(true);
        setError(null);
        const res = await apiService.get(
          `/weather/forecast?location=${encodeURIComponent(location)}&days=7`,
        );
        setWeather(res.data);
      } catch (err) {
        setError(err.message || "Failed to load weather data");
      } finally {
        setLoading(false);
        setRefreshing(false);
      }
    },
    [location],
  );

  useEffect(() => {
    fetchWeather();
  }, [fetchWeather]);

  /* ── Loading ── */
  if (loading) {
    return (
      <SafeAreaView style={styles.safe}>
        <View style={styles.centered}>
          <ActivityIndicator size="large" color={COLORS.primary} />
          <Text style={styles.loadingText}>
            Fetching weather for {location}…
          </Text>
        </View>
      </SafeAreaView>
    );
  }

  /* ── Error ── */
  if (error) {
    return (
      <SafeAreaView style={styles.safe}>
        <View style={styles.centered}>
          <Text style={styles.errorEmoji}>🌩️</Text>
          <Text style={styles.errorTitle}>Could not load weather</Text>
          <Text style={styles.errorMsg}>{error}</Text>
          <TouchableOpacity
            style={styles.retryBtn}
            onPress={() => fetchWeather()}
            activeOpacity={0.8}
          >
            <Text style={styles.retryText}>Try Again</Text>
          </TouchableOpacity>
        </View>
      </SafeAreaView>
    );
  }

  const { current, forecast, alerts } = weather;
  const uv = getUVLabel(current.uv);
  const today = forecast[0];

  return (
    <SafeAreaView style={styles.safe}>
      <StatusBar
        barStyle="light-content"
        backgroundColor={COLORS.primaryDark}
      />
      <ScrollView
        style={styles.scroll}
        contentContainerStyle={styles.content}
        showsVerticalScrollIndicator={false}
        refreshControl={
          <RefreshControl
            refreshing={refreshing}
            onRefresh={() => fetchWeather(true)}
            tintColor={COLORS.primaryLight}
          />
        }
      >
        {/* ── Hero current weather ── */}
        <View style={styles.hero}>
          <View style={styles.heroTop}>
            <View>
              <Text style={styles.heroLocation}>{weather.location.name}</Text>
              <Text style={styles.heroRegion}>{weather.location.region}</Text>
            </View>
            <Text style={styles.heroTime}>
              {new Date(weather.location.localtime).toLocaleTimeString(
                "en-IN",
                {
                  hour: "2-digit",
                  minute: "2-digit",
                },
              )}
            </Text>
          </View>

          <View style={styles.heroCenter}>
            <Text style={styles.heroEmoji}>
              {getWeatherEmoji(current.condition)}
            </Text>
            <View>
              <Text style={styles.heroTemp}>
                {Math.round(current.temp_c)}°C
              </Text>
              <Text style={styles.heroFeels}>
                Feels like {Math.round(current.feelslike_c)}°C
              </Text>
              <Text style={styles.heroCondition}>{current.condition}</Text>
            </View>
          </View>

          <View style={styles.heroStats}>
            <View style={styles.heroStat}>
              <Text style={styles.heroStatVal}>{current.humidity}%</Text>
              <Text style={styles.heroStatLabel}>Humidity</Text>
            </View>
            <View style={styles.heroStatDivider} />
            <View style={styles.heroStat}>
              <Text style={styles.heroStatVal}>{current.wind_kph} km/h</Text>
              <Text style={styles.heroStatLabel}>Wind {current.wind_dir}</Text>
            </View>
            <View style={styles.heroStatDivider} />
            <View style={styles.heroStat}>
              <Text style={styles.heroStatVal}>{current.precip_mm}mm</Text>
              <Text style={styles.heroStatLabel}>Rainfall</Text>
            </View>
          </View>
        </View>

        {/* ── Alerts ── */}
        {alerts?.length > 0 ? (
          <View style={styles.section}>
            {alerts.map((a, i) => (
              <AlertBanner key={i} alert={a} />
            ))}
          </View>
        ) : null}

        {/* ── Today's farming stats ── */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>🌾 Farming Conditions Today</Text>
          <View style={styles.statGrid}>
            <StatTile emoji="☀️" label="UV Index" value={current.uv} unit="" />
            <StatTile
              emoji="🌅"
              label="Sunrise"
              value={today.sunrise}
              unit=""
            />
            <StatTile emoji="🌇" label="Sunset" value={today.sunset} unit="" />
            <StatTile
              emoji="💧"
              label="Rain Today"
              value={today.total_precip_mm}
              unit="mm"
            />
            <StatTile
              emoji="💨"
              label="Max Wind"
              value={Math.round(today.max_wind_kph)}
              unit=" km/h"
            />
            <StatTile
              emoji="🌡️"
              label="Avg Temp"
              value={Math.round(today.avg_temp_c)}
              unit="°C"
            />
          </View>

          {/* UV bar */}
          <View style={styles.uvRow}>
            <Text style={styles.uvLabel}>UV Index: </Text>
            <View
              style={[
                styles.uvBadge,
                { backgroundColor: uv.color + "22", borderColor: uv.color },
              ]}
            >
              <Text style={[styles.uvBadgeText, { color: uv.color }]}>
                {uv.label} ({current.uv})
              </Text>
            </View>
          </View>
        </View>

        {/* ── 7-Day Forecast ── */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>📅 7-Day Forecast</Text>
          <ScrollView
            horizontal
            showsHorizontalScrollIndicator={false}
            style={styles.forecastScroll}
          >
            {forecast.map((day, i) => (
              <ForecastDay key={day.date} day={day} isToday={i === 0} />
            ))}
          </ScrollView>
        </View>

        <View style={{ height: 32 }} />
      </ScrollView>
    </SafeAreaView>
  );
};

/* ─── Styles ──────────────────────────────────────────────────────────────── */

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: COLORS.primaryDark },
  scroll: { flex: 1, backgroundColor: COLORS.background },
  content: { paddingBottom: 24 },

  centered: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    padding: 32,
  },
  loadingText: {
    marginTop: 16,
    fontSize: 14,
    color: COLORS.textSecondary,
    textAlign: "center",
  },
  errorEmoji: { fontSize: 52, marginBottom: 16 },
  errorTitle: {
    fontSize: 18,
    fontWeight: "700",
    color: COLORS.textPrimary,
    marginBottom: 8,
  },
  errorMsg: {
    fontSize: 14,
    color: COLORS.textSecondary,
    textAlign: "center",
    marginBottom: 24,
  },
  retryBtn: {
    backgroundColor: COLORS.primary,
    borderRadius: 12,
    paddingHorizontal: 28,
    paddingVertical: 14,
  },
  retryText: { color: "#fff", fontWeight: "700", fontSize: 15 },

  /* Hero */
  hero: {
    backgroundColor: COLORS.primaryDark,
    paddingHorizontal: 20,
    paddingTop: 20,
    paddingBottom: 28,
  },
  heroTop: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "flex-start",
    marginBottom: 20,
  },
  heroLocation: { fontSize: 22, fontWeight: "800", color: "#fff" },
  heroRegion: { fontSize: 13, color: "rgba(255,255,255,0.65)", marginTop: 2 },
  heroTime: { fontSize: 13, color: "rgba(255,255,255,0.65)" },

  heroCenter: {
    flexDirection: "row",
    alignItems: "center",
    gap: 16,
    marginBottom: 24,
  },
  heroEmoji: { fontSize: 64 },
  heroTemp: { fontSize: 52, fontWeight: "800", color: "#fff", lineHeight: 56 },
  heroFeels: { fontSize: 13, color: "rgba(255,255,255,0.7)", marginTop: 2 },
  heroCondition: {
    fontSize: 16,
    fontWeight: "600",
    color: "rgba(255,255,255,0.9)",
    marginTop: 4,
  },

  heroStats: {
    flexDirection: "row",
    backgroundColor: "rgba(255,255,255,0.12)",
    borderRadius: 14,
    padding: 14,
  },
  heroStat: { flex: 1, alignItems: "center" },
  heroStatVal: { fontSize: 15, fontWeight: "700", color: "#fff" },
  heroStatLabel: {
    fontSize: 11,
    color: "rgba(255,255,255,0.65)",
    marginTop: 3,
  },
  heroStatDivider: {
    width: 1,
    backgroundColor: "rgba(255,255,255,0.2)",
    marginHorizontal: 4,
  },

  /* Alerts */
  alertBanner: {
    flexDirection: "row",
    backgroundColor: "#FFF3E0",
    borderRadius: 12,
    padding: 14,
    marginBottom: 10,
    borderLeftWidth: 4,
    borderLeftColor: COLORS.secondary,
    gap: 12,
    alignItems: "flex-start",
  },
  alertIcon: { fontSize: 22 },
  alertText: { flex: 1 },
  alertTitle: {
    fontSize: 14,
    fontWeight: "700",
    color: COLORS.textPrimary,
    marginBottom: 4,
  },
  alertDesc: { fontSize: 12, color: COLORS.textSecondary },

  /* Sections */
  section: { padding: 20, paddingBottom: 0 },
  sectionTitle: {
    fontSize: 15,
    fontWeight: "700",
    color: COLORS.textPrimary,
    marginBottom: 14,
  },

  /* Stat grid */
  statGrid: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 10,
    marginBottom: 14,
  },
  statTile: {
    width: "30%",
    flexGrow: 1,
    backgroundColor: COLORS.surface,
    borderRadius: 12,
    padding: 12,
    alignItems: "center",
    ...SHADOW.sm,
  },
  statTileEmoji: { fontSize: 20, marginBottom: 6 },
  statTileValue: { fontSize: 16, fontWeight: "700", color: COLORS.textPrimary },
  statTileUnit: { fontSize: 11, fontWeight: "500", color: COLORS.textMuted },
  statTileLabel: {
    fontSize: 10,
    color: COLORS.textMuted,
    marginTop: 3,
    textAlign: "center",
  },

  /* UV */
  uvRow: { flexDirection: "row", alignItems: "center", marginBottom: 4 },
  uvLabel: { fontSize: 13, fontWeight: "600", color: COLORS.textSecondary },
  uvBadge: {
    borderRadius: 20,
    borderWidth: 1,
    paddingHorizontal: 10,
    paddingVertical: 3,
  },
  uvBadgeText: { fontSize: 12, fontWeight: "700" },

  /* Forecast */
  forecastScroll: { marginHorizontal: -4 },
  forecastCard: {
    width: 110,
    backgroundColor: COLORS.surface,
    borderRadius: 16,
    padding: 14,
    marginHorizontal: 5,
    alignItems: "center",
    ...SHADOW.sm,
  },
  forecastCardToday: {
    backgroundColor: COLORS.primary,
  },
  forecastDay: {
    fontSize: 13,
    fontWeight: "700",
    color: COLORS.textSecondary,
    marginBottom: 2,
  },
  forecastDayToday: { color: "rgba(255,255,255,0.8)" },
  forecastDate: { fontSize: 11, color: COLORS.textMuted, marginBottom: 10 },
  forecastEmoji: { fontSize: 28, marginBottom: 8 },
  forecastCondition: {
    fontSize: 11,
    color: COLORS.textSecondary,
    textAlign: "center",
    marginBottom: 10,
  },
  forecastConditionToday: { color: "rgba(255,255,255,0.8)" },
  forecastTemps: { flexDirection: "row", gap: 6, marginBottom: 6 },
  forecastMax: { fontSize: 16, fontWeight: "800", color: COLORS.textPrimary },
  forecastMin: { fontSize: 16, fontWeight: "500", color: COLORS.textMuted },
  forecastRain: { fontSize: 11, color: "#0288D1", fontWeight: "600" },
});

export default WeatherScreen;
