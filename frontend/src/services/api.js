// src/services/api.js
import API_CONFIG from "../config/api.js";

class ApiService {
  constructor() {
    this.baseURL = API_CONFIG.baseURL;
  }

  async request(endpoint, options = {}) {
    const url = `${this.baseURL}${endpoint}`;
    const config = {
      headers: {
        "Content-Type": "application/json",
        ...options.headers,
      },
      ...options,
    };

    // Add auth token if available
    const token = localStorage.getItem("token");
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }

    try {
      const response = await fetch(url, config);

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error("API request failed:", error);
      throw error;
    }
  }

  // Auth methods
  async login(credentials) {
    return this.request(API_CONFIG.endpoints.auth.login, {
      method: "POST",
      body: JSON.stringify(credentials),
    });
  }

  async signup(userData) {
    return this.request(API_CONFIG.endpoints.auth.signup, {
      method: "POST",
      body: JSON.stringify(userData),
    });
  }

  // Onboarding
  async submitOnboarding(data) {
    return this.request(API_CONFIG.endpoints.onboarding, {
      method: "POST",
      body: JSON.stringify(data),
    });
  }

  // Simulations
  async getSimulations() {
    return this.request(API_CONFIG.endpoints.simulations);
  }

  async getSimulationById(id) {
    return this.request(`${API_CONFIG.endpoints.simulations}/${id}`);
  }

  // Instruments
  async getInstruments(limit = 1000) {
    return this.request(`${API_CONFIG.endpoints.instruments}?limit=${limit}`);
  }

  async getBulkInstruments(symbols) {
    return this.request(`${API_CONFIG.endpoints.instruments}/bulk`, {
      method: "POST",
      body: JSON.stringify({ symbols }),
    });
  }

  // Users
  async getUserById(userId) {
    return this.request(`${API_CONFIG.endpoints.users}/${userId}`);
  }
}

export default new ApiService();
