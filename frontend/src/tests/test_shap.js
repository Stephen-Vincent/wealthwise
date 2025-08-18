// test_shap.js - Frontend SHAP Testing Utility
// Save this as a separate file in your frontend project and import it for testing

/**
 * 🧪 Comprehensive SHAP Testing Suite for Frontend
 *
 * Usage:
 * 1. Save this file as test_shap.js in your frontend project
 * 2. Import it: import { testShapData, monitorShapRequests } from './test_shap.js'
 * 3. Run tests in your console or component
 */

// =============================================================================
// MAIN SHAP TESTING FUNCTIONS
// =============================================================================

/**
 * 🔍 Comprehensive SHAP Data Analysis
 * Tests a portfolio data object for SHAP content
 */
export function testShapData(portfolioData, simulationId = null) {
  console.log("🧪 Starting comprehensive SHAP data test...");
  console.log(
    "📊 Testing simulation:",
    simulationId || portfolioData?.id || "unknown"
  );

  if (!portfolioData) {
    console.log("❌ No portfolio data provided");
    return { error: "No portfolio data provided" };
  }

  const results = {
    simulationId: simulationId || portfolioData.id,
    timestamp: new Date().toISOString(),
    hasShapData: false,
    shapLocation: null,
    shapAnalysis: {},
    issues: [],
    recommendations: [],
  };

  try {
    // Check data structure
    console.log("📋 Portfolio data keys:", Object.keys(portfolioData));

    // 1. Check top-level SHAP data
    console.log("\n🔍 Checking top-level SHAP data...");
    if (portfolioData.shap_explanation) {
      console.log("✅ Found SHAP data at top level!");
      results.hasShapData = true;
      results.shapLocation = "top-level";
      results.shapAnalysis = analyzeShapObject(
        portfolioData.shap_explanation,
        "top-level"
      );
    } else {
      console.log("❌ No SHAP data at top level");
    }

    // 2. Check results-level SHAP data
    console.log("\n🔍 Checking results-level SHAP data...");
    if (portfolioData.results?.shap_explanation) {
      console.log("✅ Found SHAP data in results!");
      if (!results.hasShapData) {
        results.hasShapData = true;
        results.shapLocation = "results";
        results.shapAnalysis = analyzeShapObject(
          portfolioData.results.shap_explanation,
          "results"
        );
      } else {
        console.log("⚠️ SHAP data found in multiple locations!");
        results.issues.push("SHAP data exists in multiple locations");
      }
    } else {
      console.log("❌ No SHAP data in results");
    }

    // 3. Check SHAP flags
    console.log("\n🏁 Checking SHAP flags...");
    const wealthwiseEnhanced = portfolioData.wealthwise_enhanced;
    const hasShapFlag = portfolioData.has_shap_explanations;
    const methodology = portfolioData.methodology;

    console.log("  wealthwise_enhanced:", wealthwiseEnhanced);
    console.log("  has_shap_explanations:", hasShapFlag);
    console.log("  methodology:", methodology);

    results.flags = {
      wealthwiseEnhanced,
      hasShapFlag,
      methodology,
    };

    // 4. Check for inconsistencies
    console.log("\n⚠️ Checking for inconsistencies...");
    if (hasShapFlag && !results.hasShapData) {
      const issue = "has_shap_explanations=true but no SHAP data found";
      console.log("⚠️ ISSUE:", issue);
      results.issues.push(issue);
    }

    if (wealthwiseEnhanced && !results.hasShapData) {
      const issue = "wealthwise_enhanced=true but no SHAP data found";
      console.log("⚠️ ISSUE:", issue);
      results.issues.push(issue);
    }

    // 5. Deep search for hidden SHAP data
    if (!results.hasShapData) {
      console.log("\n🔍 Performing deep search for hidden SHAP data...");
      const deepFindings = deepSearchForShap(portfolioData);
      if (deepFindings.length > 0) {
        console.log("🔍 Found SHAP-related data in unexpected locations:");
        deepFindings.forEach((finding) => {
          console.log(`  📍 ${finding.path}:`, finding.type);
        });
        results.hiddenShapData = deepFindings;
      } else {
        console.log("🔍 No hidden SHAP data found");
      }
    }

    // 6. Generate recommendations
    generateRecommendations(results);

    // 7. Summary
    console.log("\n🎯 === SHAP TEST SUMMARY ===");
    if (results.hasShapData) {
      console.log("✅ SHAP data IS available!");
      console.log(`📍 Location: ${results.shapLocation}`);
      console.log("🔧 Access via:", getShapAccessPath(results.shapLocation));
    } else {
      console.log("❌ SHAP data is NOT available");
      console.log("🔧 This explains why your SHAP dashboard is not visible");
    }

    if (results.issues.length > 0) {
      console.log("⚠️ Issues found:", results.issues.length);
      results.issues.forEach((issue) => console.log(`  - ${issue}`));
    }

    if (results.recommendations.length > 0) {
      console.log("💡 Recommendations:");
      results.recommendations.forEach((rec) => console.log(`  - ${rec}`));
    }
  } catch (error) {
    console.error("❌ SHAP test failed:", error);
    results.error = error.message;
  }

  // Store results for inspection
  window.lastShapTestResults = results;
  console.log("\n💾 Results stored in window.lastShapTestResults");

  return results;
}

/**
 * 📊 Analyze a SHAP explanation object
 */
function analyzeShapObject(shapObj, location) {
  console.log(`📊 Analyzing SHAP object at ${location}...`);

  const analysis = {
    location,
    keys: Object.keys(shapObj),
    portfolioQualityScore: shapObj.portfolio_quality_score,
    hasHumanReadable: !!shapObj.human_readable_explanation,
    hasFeatureImportance: !!shapObj.feature_importance,
    hasShapValues: !!shapObj.shap_values,
    components: {},
  };

  console.log(`  📋 SHAP keys (${analysis.keys.length}):`, analysis.keys);

  // Analyze each component
  if (analysis.portfolioQualityScore !== undefined) {
    console.log(
      `  💯 Portfolio Quality Score: ${analysis.portfolioQualityScore}`
    );
    analysis.components.qualityScore = analysis.portfolioQualityScore;
  }

  if (analysis.hasHumanReadable) {
    const explanations = shapObj.human_readable_explanation;
    const explanationKeys = Object.keys(explanations);
    console.log(
      `  📝 Human Readable Explanations (${explanationKeys.length}):`,
      explanationKeys
    );
    analysis.components.humanReadable = explanationKeys;

    // Show first explanation as example
    if (explanationKeys.length > 0) {
      const firstKey = explanationKeys[0];
      console.log(`    Example - ${firstKey}: "${explanations[firstKey]}"`);
    }
  }

  if (analysis.hasFeatureImportance) {
    const importance = shapObj.feature_importance;
    const importanceKeys = Object.keys(importance);
    console.log(
      `  ⚖️ Feature Importance (${importanceKeys.length}):`,
      importanceKeys
    );
    analysis.components.featureImportance = importanceKeys;

    // Show example values
    importanceKeys.slice(0, 3).forEach((key) => {
      console.log(`    ${key}: ${importance[key]}`);
    });
  }

  if (analysis.hasShapValues) {
    const values = shapObj.shap_values;
    console.log(
      `  📈 SHAP Values: ${
        Array.isArray(values) ? `Array[${values.length}]` : typeof values
      }`
    );
    analysis.components.shapValues = Array.isArray(values)
      ? values.length
      : typeof values;
  }

  return analysis;
}

/**
 * 🔍 Deep search for SHAP-related data
 */
function deepSearchForShap(obj, path = "", visited = new Set(), maxDepth = 5) {
  if (!obj || typeof obj !== "object" || visited.has(obj) || maxDepth <= 0) {
    return [];
  }

  visited.add(obj);
  const findings = [];

  try {
    Object.entries(obj).forEach(([key, value]) => {
      const currentPath = path ? `${path}.${key}` : key;

      // Check for SHAP-related keys
      if (
        key.toLowerCase().includes("shap") ||
        key.toLowerCase().includes("explanation") ||
        key.toLowerCase().includes("quality") ||
        key.toLowerCase().includes("feature_importance")
      ) {
        findings.push({
          path: currentPath,
          key: key,
          type: Array.isArray(value) ? `array[${value.length}]` : typeof value,
          value: value,
        });
      }

      // Recurse into objects
      if (value && typeof value === "object") {
        findings.push(
          ...deepSearchForShap(value, currentPath, visited, maxDepth - 1)
        );
      }
    });
  } catch (error) {
    // Handle circular references or other issues
  }

  return findings;
}

/**
 * 💡 Generate recommendations based on test results
 */
function generateRecommendations(results) {
  results.recommendations = [];

  if (!results.hasShapData) {
    if (results.flags.wealthwiseEnhanced) {
      results.recommendations.push(
        "Backend is generating SHAP data but not exposing it in API response"
      );
      results.recommendations.push(
        "Check format_enhanced_simulation_response() function"
      );
      results.recommendations.push(
        "Ensure SHAP data is included at top-level in API response"
      );
    } else {
      results.recommendations.push(
        "WealthWise enhanced features are not enabled"
      );
      results.recommendations.push(
        "Check if WealthWise system is properly initialized"
      );
    }
  }

  if (results.shapLocation === "results") {
    results.recommendations.push("SHAP data is buried in results object");
    results.recommendations.push(
      "Update frontend to check portfolioData.results.shap_explanation"
    );
    results.recommendations.push(
      "Consider exposing SHAP data at top level for easier access"
    );
  }

  if (results.issues.length > 0) {
    results.recommendations.push(
      "Fix flag inconsistencies between has_shap_explanations and actual data"
    );
  }

  if (results.hiddenShapData?.length > 0) {
    results.recommendations.push(
      "SHAP data found in unexpected locations - check data structure"
    );
  }
}

/**
 * 🔗 Get access path for SHAP data
 */
function getShapAccessPath(location) {
  switch (location) {
    case "top-level":
      return "portfolioData.shap_explanation";
    case "results":
      return "portfolioData.results.shap_explanation";
    default:
      return "Not found";
  }
}

// =============================================================================
// NETWORK MONITORING FUNCTIONS
// =============================================================================

/**
 * 🕵️ Monitor API requests for SHAP data
 */
export function monitorShapRequests() {
  console.log("🕵️ Starting SHAP request monitoring...");

  if (window._shapMonitorActive) {
    console.log("⚠️ Monitoring already active");
    return;
  }

  // Store original fetch
  const originalFetch = window.fetch;
  window._originalFetch = originalFetch;
  window._shapMonitorActive = true;

  // Intercept fetch requests
  window.fetch = function (...args) {
    const [url, options = {}] = args;

    // Check if this is a simulation-related request
    if (url.includes("simulation") || url.includes("portfolio")) {
      console.log("\n📡 === SIMULATION API REQUEST ===");
      console.log("🔗 URL:", url);
      console.log("🔧 Method:", options.method || "GET");

      // Call original fetch and analyze response
      return originalFetch.apply(this, args).then((response) => {
        console.log("📥 Status:", response.status);

        if (response.ok) {
          // Clone response to read without consuming
          const clonedResponse = response.clone();

          clonedResponse
            .json()
            .then((data) => {
              console.log("📊 Analyzing response for SHAP data...");
              const testResults = testShapData(data, extractSimulationId(url));

              // Store for inspection
              window.lastApiResponse = data;
              window.lastShapTest = testResults;
            })
            .catch((error) => {
              console.log("❌ Failed to parse JSON response:", error.message);
            });
        } else {
          console.log("❌ Request failed:", response.status);
        }

        return response;
      });
    }

    // Pass through non-simulation requests
    return originalFetch.apply(this, args);
  };

  console.log("✅ SHAP monitoring active");
  console.log("💡 Navigate to your portfolio page to see analysis");
  console.log("💡 Stop with: stopShapMonitoring()");
}

/**
 * 🛑 Stop monitoring API requests
 */
export function stopShapMonitoring() {
  if (window._originalFetch) {
    window.fetch = window._originalFetch;
    delete window._originalFetch;
    window._shapMonitorActive = false;
    console.log("✅ SHAP monitoring stopped");
  } else {
    console.log("⚠️ No monitoring to stop");
  }
}

/**
 * 📊 Extract simulation ID from URL
 */
function extractSimulationId(url) {
  const match = url.match(/simulation[s]?\/(\d+)/i);
  return match ? match[1] : null;
}

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

/**
 * 📦 Test stored portfolio data
 */
export function testStoredPortfolioData() {
  console.log("📦 Testing stored portfolio data...");

  try {
    const stored = localStorage.getItem("portfolioData");
    if (stored) {
      const data = JSON.parse(stored);
      console.log("✅ Found stored portfolio data");
      return testShapData(data, data.id);
    } else {
      console.log("❌ No stored portfolio data found");
      return { error: "No stored data" };
    }
  } catch (error) {
    console.log("❌ Failed to parse stored data:", error.message);
    return { error: error.message };
  }
}

/**
 * 🎯 Quick SHAP check
 */
export function quickShapCheck(portfolioData) {
  if (!portfolioData) {
    console.log("❌ No portfolio data provided");
    return false;
  }

  const hasTopLevel = !!portfolioData.shap_explanation;
  const hasResults = !!portfolioData.results?.shap_explanation;
  const hasShap = hasTopLevel || hasResults;

  console.log("🎯 Quick SHAP Check:");
  console.log("  Top-level SHAP:", hasTopLevel);
  console.log("  Results SHAP:", hasResults);
  console.log("  WealthWise enhanced:", portfolioData.wealthwise_enhanced);
  console.log("  Has SHAP flag:", portfolioData.has_shap_explanations);

  return {
    hasShap,
    location: hasTopLevel ? "top-level" : hasResults ? "results" : null,
    accessPath: hasTopLevel
      ? "portfolioData.shap_explanation"
      : hasResults
      ? "portfolioData.results.shap_explanation"
      : null,
  };
}

/**
 * 🔧 Mock SHAP data for testing
 */
export function createMockShapData() {
  return {
    portfolio_quality_score: 78.5,
    confidence_score: 85.2,
    human_readable_explanation: {
      risk_score:
        "Your risk tolerance of 65/100 supports growth-oriented investments",
      timeframe: "Your 5-year timeline allows for market volatility",
      target_value: "Your £20,000 target requires moderate growth strategy",
      market_conditions: "Current bull market supports equity allocation",
    },
    feature_importance: {
      risk_tolerance: 0.32,
      time_horizon: 0.28,
      target_amount: 0.25,
      market_volatility: 0.15,
    },
    shap_values: [0.32, 0.28, 0.25, 0.15],
    expected_value: 0.67,
    methodology: "SHAP TreeExplainer with XGBoost model",
  };
}

/**
 * 🧪 Test with mock data
 */
export function testWithMockData() {
  console.log("🧪 Testing SHAP dashboard with mock data...");

  const mockPortfolioData = {
    id: 999,
    wealthwise_enhanced: true,
    has_shap_explanations: true,
    methodology: "WealthWise SHAP-enhanced optimization",
    shap_explanation: createMockShapData(),
    results: {
      // ... other results data
    },
  };

  return testShapData(mockPortfolioData, 999);
}

// =============================================================================
// GLOBAL FUNCTIONS FOR CONSOLE USE
// =============================================================================

// Make functions available globally for console testing
if (typeof window !== "undefined") {
  window.testShapData = testShapData;
  window.monitorShapRequests = monitorShapRequests;
  window.stopShapMonitoring = stopShapMonitoring;
  window.testStoredPortfolioData = testStoredPortfolioData;
  window.quickShapCheck = quickShapCheck;
  window.testWithMockData = testWithMockData;

  console.log("🧪 SHAP Testing Suite loaded!");
  console.log("💡 Available functions:");
  console.log("  - testShapData(portfolioData)");
  console.log("  - monitorShapRequests()");
  console.log("  - testStoredPortfolioData()");
  console.log("  - quickShapCheck(portfolioData)");
  console.log("  - testWithMockData()");
}

// =============================================================================
// EXPORT FOR MODULE USE
// =============================================================================

export default {
  testShapData,
  monitorShapRequests,
  stopShapMonitoring,
  testStoredPortfolioData,
  quickShapCheck,
  createMockShapData,
  testWithMockData,
};
