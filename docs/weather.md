# Weather Module

The Abi AI framework now includes a comprehensive weather module that provides real-time weather data fetching capabilities using the OpenWeatherMap API.

## Features

- **Current Weather**: Get current weather conditions for any city
- **5-Day Forecast**: Retrieve detailed 5-day weather forecasts
- **Coordinate-based Weather**: Get weather data using latitude/longitude coordinates
- **Multiple Units**: Support for metric, imperial, and Kelvin temperature units
- **Multi-language**: Weather descriptions in multiple languages
- **Web Interface**: Beautiful HTML interface for easy weather access
- **REST API**: Full REST API endpoints for programmatic access

## Setup

### 1. Get OpenWeatherMap API Key

1. Visit [OpenWeatherMap](https://openweathermap.org/)
2. Sign up for a free account
3. Get your API key from the dashboard
4. The free tier allows 1000 calls per day

### 2. Configure API Key

You need to replace `"YOUR_API_KEY_HERE"` in the following files:

- `src/web_server.zig` (in weather handlers)
- `examples/weather_client.zig` (when running the example)

## Usage

### Command Line Example

```bash
# Build the weather client example
zig build -Dtarget=x86_64-linux -Doptimize=Release

# Run with your API key and city
./zig-out/bin/weather_client YOUR_API_KEY London
```

### Web Interface

1. Start the web server:
```bash
zig build run
```

2. Open your browser and navigate to:
```
http://localhost:3000/weather
```

3. Enter a city name and click "Get Weather"

### REST API Endpoints

#### Current Weather
```
GET /api/weather/current?city=London
```

Response:
```json
{
  "temperature": 15.5,
  "feels_like": 14.2,
  "humidity": 65,
  "pressure": 1013,
  "description": "scattered clouds",
  "icon": "03d",
  "wind_speed": 3.2,
  "wind_direction": 180,
  "visibility": 10000,
  "sunrise": 1640995200,
  "sunset": 1641027600,
  "city": "London",
  "country": "GB",
  "timestamp": 1641009600
}
```

#### 5-Day Forecast
```
GET /api/weather/forecast?city=London
```

#### Weather by Coordinates
```
GET /api/weather/coords?lat=51.5074&lon=-0.1278
```

## API Reference

### WeatherData Structure

```zig
pub const WeatherData = struct {
    temperature: f32,        // Temperature in configured units
    feels_like: f32,        // "Feels like" temperature
    humidity: u8,           // Humidity percentage
    pressure: u16,          // Atmospheric pressure in hPa
    description: []const u8, // Weather description
    icon: []const u8,       // Weather icon code
    wind_speed: f32,        // Wind speed in m/s
    wind_direction: u16,    // Wind direction in degrees
    visibility: u32,        // Visibility in meters
    sunrise: u64,           // Sunrise timestamp
    sunset: u64,            // Sunset timestamp
    city: []const u8,       // City name
    country: []const u8,    // Country code
    timestamp: u64,         // Data timestamp
};
```

### WeatherService

```zig
// Initialize weather service
var weather_service = try WeatherService.init(allocator, config);
defer weather_service.deinit();

// Get current weather by city
const weather = try weather_service.getCurrentWeather("London");

// Get current weather by coordinates
const weather = try weather_service.getCurrentWeatherByCoords(51.5074, -0.1278);

// Get 5-day forecast
const forecast = try weather_service.getForecast("London");
```

### WeatherUtils

```zig
// Temperature conversions
const celsius = WeatherUtils.kelvinToCelsius(273.15);
const fahrenheit = WeatherUtils.celsiusToFahrenheit(0.0);

// Wind direction
const direction = WeatherUtils.getWindDirection(180); // Returns "S"

// Weather emoji
const emoji = WeatherUtils.getWeatherEmoji("03d"); // Returns "☁️"
```

## Configuration

### WeatherConfig

```zig
pub const WeatherConfig = struct {
    api_key: []const u8,           // OpenWeatherMap API key
    base_url: []const u8 = "https://api.openweathermap.org/data/2.5",
    units: []const u8 = "metric",  // metric, imperial, kelvin
    language: []const u8 = "en",   // Language for descriptions
    timeout_seconds: u32 = 10,     // Request timeout
};
```

## Error Handling

The weather module provides comprehensive error handling:

- `WeatherApiError`: API request failed
- `InvalidResponse`: Invalid JSON response from API
- Network errors are properly propagated

## Examples

### Basic Weather Client

```zig
const std = @import("std");
const root = @import("root.zig");

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const config = root.weather.WeatherConfig{
        .api_key = "YOUR_API_KEY",
        .units = "metric",
    };

    var service = try root.weather.WeatherService.init(allocator, config);
    defer service.deinit();

    const weather = try service.getCurrentWeather("Tokyo");
    defer weather.deinit(allocator);

    std.debug.print("Temperature: {d}°C\n", .{weather.temperature});
    std.debug.print("Description: {s}\n", .{weather.description});
}
```

### Web Server Integration

The weather module is fully integrated with the web server. Simply start the server and access the weather endpoints:

```bash
zig build run
```

Then visit `http://localhost:3000/weather` for the web interface.

## Performance

- **HTTP Client**: Uses Zig's built-in HTTP client for efficient requests
- **Memory Management**: Proper memory cleanup with arena allocators
- **Error Handling**: Comprehensive error handling with proper resource cleanup
- **JSON Parsing**: Efficient JSON parsing using Zig's standard library

## Dependencies

- **OpenWeatherMap API**: Free weather data service
- **Zig Standard Library**: HTTP client, JSON parsing, memory management
- **No external dependencies**: Pure Zig implementation

## Contributing

To add new weather features:

1. Extend the `WeatherData` structure
2. Add new methods to `WeatherService`
3. Update the web server handlers
4. Add tests for new functionality
5. Update documentation

## License

This weather module is part of the Abi AI framework and follows the same license terms. 