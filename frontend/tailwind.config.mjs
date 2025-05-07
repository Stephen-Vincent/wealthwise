/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["'Cabin Condensed'", "sans-serif"],
      },
      colors: {
        primary: "#00A8FF",
        secondary: "#EAF6FB",
      },
      borderRadius: {
        btn: "15px",
      },
    },
  },
  plugins: [],
};
