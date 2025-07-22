// src/config/api.js
const API_CONFIG = {
  baseURL: import.meta.env.VITE_API_URL || "http://localhost:8000",
  apiURL: import.meta.env.VITE_API_BASE_URL || "http://localhost:8000/api",
  endpoints: {
    auth: {
      login: "/auth/login",
      signup: "/auth/signup",
      logout: "/auth/logout",
    },
    onboarding: "/onboarding/",
    simulations: "/simulations",
    instruments: "/api/instruments",
    users: "/api/users",
  },
};

export default API_CONFIG;
