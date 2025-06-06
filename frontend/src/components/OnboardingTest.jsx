import React from "react";
import axios from "axios";

function OnboardingTest() {
  const handleSubmit = async () => {
    const payload = {
      name: "Stephen Vincent",
      goal: "buy a horse",
      target_value: 12450,
    };

    try {
      const response = await axios.post(
        "http://127.0.0.1:5000/onboarding",
        payload
      );
      console.log("Success:", response.data);
      alert(`Success: ${response.data.message}`);
    } catch (error) {
      console.error(
        "Error:",
        error.response ? error.response.data : error.message
      );
      alert("Error submitting onboarding data");
    }
  };

  return (
    <div>
      <h2>Onboarding API Test</h2>
      <button onClick={handleSubmit}>Send Test Onboarding Data</button>
    </div>
  );
}

export default OnboardingTest;
