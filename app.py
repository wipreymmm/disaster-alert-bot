from flask import Flask, request, jsonify, render_template
from model.rag_modelv4 import ask_question, refresh_web_data
from datetime import datetime
import logging
import os
import requests
from dotenv import load_dotenv

# --- App setup ---
app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.jinja_env.auto_reload = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

load_dotenv()
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

# Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.context_processor
def inject_now():
    return {'now': datetime.now()}

# Geocoding to pass it to open meteo
def geocode(city_name):
    """
    Try OpenWeather geocoding if API key present, otherwise fall back to Nominatim.
    Returns (lat, lon, display_name) or None on failure.
    """
    try:
        if OPENWEATHER_API_KEY:
            url = f"http://api.openweathermap.org/geo/1.0/direct?q={requests.utils.requote_uri(city_name)}&limit=1&appid={OPENWEATHER_API_KEY}"
            r = requests.get(url, timeout=6)
            arr = r.json()
            if isinstance(arr, list) and len(arr) > 0:
                lat = arr[0].get("lat")
                lon = arr[0].get("lon")
                name = arr[0].get("name") or city_name
                state = arr[0].get("state")
                country = arr[0].get("country")
                display = f"{name}" + (f", {state}" if state else "") + (f", {country}" if country else "")
                return float(lat), float(lon), display
        # fallback api in case the first one fail
        nom_url = f"https://nominatim.openstreetmap.org/search?q={requests.utils.requote_uri(city_name)}&format=json&limit=1"
        r = requests.get(nom_url, headers={"User-Agent": "DisasterAlertBot/1.0 (youremail@example.com)"}, timeout=6)
        arr = r.json()
        if isinstance(arr, list) and len(arr) > 0:
            lat = arr[0].get("lat")
            lon = arr[0].get("lon")
            display = arr[0].get("display_name", city_name)
            return float(lat), float(lon), display
    except Exception:
        app.logger.exception("Geocode failed")
    return None


# ---------- Basic routes ----------
@app.route('/', endpoint='index')
def index():
    return render_template('index.html', title='DisasterAlertBot')


@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json()
        if not data:
            logger.error("No JSON data received")
            return jsonify({"answer": "Error: No data received"}), 400
        message = data.get("message", "").strip()
        if not message:
            logger.warning("Empty message received")
            return jsonify({"answer": "Please enter a question."}), 400
        logger.info(f"Received question: {message}")
        answer = ask_question(message)
        logger.info(f"Generated answer: {answer[:100]}...")
        return jsonify({"answer": answer}), 200
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}", exc_info=True)
        return jsonify({"answer": "Sorry, I encountered an error. Please try again."}), 500


@app.route("/refresh", methods=["POST"])
def refresh():
    try:
        logger.info("Refreshing web data...")
        refresh_web_data()
        logger.info("Web data refreshed successfully")
        return jsonify({"status": "success", "message": "Web sources refreshed!"}), 200
    except Exception as e:
        logger.error(f"Error refreshing data: {str(e)}", exc_info=True)
        return jsonify({"status": "error", "message": f"Refresh failed: {str(e)}"}), 500


# Bootstrap icon map
WEATHERCODE_MAP = {
    0: ("Clear Sky", "bi-brightness-high"),
    1: ("Mainly Clear", "bi-brightness-high"),
    2: ("Partly Cloudy", "bi-cloud-sun"),
    3: ("Overcast", "bi-cloud"),

    45: ("Fog", "bi-cloud-fog"),
    48: ("Fog", "bi-cloud-fog"),

    51: ("Light Drizzle", "bi-cloud-drizzle"),
    53: ("Drizzle", "bi-cloud-drizzle"),
    55: ("Heavy Drizzle", "bi-cloud-drizzle"),

    61: ("Light Rain", "bi-cloud-rain"),
    63: ("Moderate Rain", "bi-cloud-rain-heavy"),
    65: ("Heavy Rain", "bi-cloud-rain-heavy"),

    80: ("Rain Showers", "bi-cloud-rain"),
    81: ("Rain Showers", "bi-cloud-rain"),
    82: ("Heavy Rain Showers", "bi-cloud-rain-heavy"),

    95: ("Thunderstorm", "bi-cloud-lightning"),
    96: ("Thunderstorm w/ Hail", "bi-cloud-lightning-rain"),
    99: ("Severe Thunderstorm", "bi-cloud-lightning-rain")
}


@app.route("/weather", methods=["GET"])
def get_weather():
    """
    Returns current weather using Open-Meteo current_weather.
    Accepts: city=Name (no lat/lon)
    Response:
      {
        "city":"Baguio City",
        "display_city":"Baguio City, Philippines",
        "temp": 18,
        "feels_like": null,
        "humidity": null,
        "wind": 1.34,
        "condition":"Broken Clouds",
        "icon":"bi-cloud",
        "provider":"Open-Meteo",
        "fetched_at":"2025-11-20T04:00:00Z"
      }
    """
    city = request.args.get("city")

    if not city:
        return jsonify({"error": "Please provide a city name via ?city=..."}), 400
    # geocode the city into lat/lon
    coords = geocode(city)
    if not coords:
        return jsonify({"error": "Unable to geocode city. Please try a different city name."}), 400

    lat, lon, display_city = coords
    try:
        # Open-Meteo: request current weather and forecast
        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            "&current_weather=true"
            "&daily=temperature_2m_max,temperature_2m_min,weathercode,precipitation_sum"
            "&timezone=Asia%2FManila"
            "&forecast_days=1"
        )
        r = requests.get(url, timeout=8)
        data = r.json()

        # read current weather
        cw = data.get("current_weather") or {}
        fetched_at = cw.get("time")

        code = cw.get("weathercode")
        cond, icon = WEATHERCODE_MAP.get(int(code) if code is not None else None, ("Unknown", "bi-cloud"))

        out = {
            "city": city,
            "display_city": display_city,
            "temp": round(cw.get("temperature")) if cw.get("temperature") is not None else None,
            "feels_like": None,
            "humidity": None,
            "wind": cw.get("windspeed"),
            "condition": cond,
            "icon": icon,
            "provider": "Open-Meteo",
            "fetched_at": fetched_at
        }
        return jsonify(out), 200

    except Exception as e:
        app.logger.exception("Open-Meteo current fetch failed")
        return jsonify({"error": str(e)}), 500


@app.route("/forecast", methods=["GET"])
def get_forecast():
    """
    Returns daily forecast (5 days) using Open-Meteo.
    Accepts: city=Name (no lat/lon)
    Returns JSON: {"daily":[{date,temp,min,max,condition,icon},...], "display_city": ...}
    """
    city = request.args.get("city")

    if not city:
        return jsonify({"error": "Please provide a city name via ?city=..."}), 400

    coords = geocode(city)
    if not coords:
        return jsonify({"error": "Unable to geocode city. Please try a different city name."}), 400

    lat, lon, display_city = coords

    try:
        om_url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            "&daily=temperature_2m_max,temperature_2m_min,weathercode,precipitation_sum"
            "&timezone=Asia%2FManila"
            "&forecast_days=7"
        )
        r = requests.get(om_url, timeout=8)
        om = r.json()

        daily = []
        d = om.get("daily", {})
        dates = d.get("time", [])
        maxes = d.get("temperature_2m_max", [])
        mins = d.get("temperature_2m_min", [])
        codes = d.get("weathercode", [])

        for i, date in enumerate(dates):
            if i >= 5:
                break
            maxv = maxes[i] if i < len(maxes) else None
            minv = mins[i] if i < len(mins) else None
            code = int(codes[i]) if i < len(codes) and codes[i] is not None else None
            cond, icon = WEATHERCODE_MAP.get(code, ("Unknown", "bi-cloud"))
            temp_rep = round(((maxv or 0) + (minv or 0)) / 2) if (maxv is not None and minv is not None) else (round(maxv) if maxv is not None else None)
            daily.append({
                "date": date,
                "temp": temp_rep,
                "min": round(minv) if minv is not None else None,
                "max": round(maxv) if maxv is not None else None,
                "condition": cond,
                "icon": icon
            })

        return jsonify({"daily": daily, "display_city": display_city}), 200

    except Exception as e:
        app.logger.exception("Open-Meteo forecast failed")
        return jsonify({"error": str(e)}), 500


# Error Handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500


# Run app
if __name__ == "__main__":
    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000
    )
