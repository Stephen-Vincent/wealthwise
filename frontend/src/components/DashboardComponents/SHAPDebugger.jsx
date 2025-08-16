import React from "react";

const SHAPDebugger = ({ portfolioData }) => {
  // Check all possible SHAP data locations
  const shapLocations = {
    "results.shap_explanation": portfolioData?.results?.shap_explanation,
    "results.shap_explanations": portfolioData?.results?.shap_explanations,
    "results.recommendation_result.shap_explanation":
      portfolioData?.results?.recommendation_result?.shap_explanation,
    "results.portfolio_recommendations.shap_explanations":
      portfolioData?.results?.portfolio_recommendations?.shap_explanations,
    full_results: portfolioData?.results,
  };

  return (
    <div className="p-6 bg-gray-50 rounded-lg">
      <h2 className="text-xl font-bold mb-4">🔍 SHAP Data Debugger</h2>

      {Object.entries(shapLocations).map(([location, data]) => (
        <div key={location} className="mb-4 p-4 bg-white rounded border">
          <h3 className="font-semibold text-gray-800 mb-2">{location}</h3>
          <div className="text-sm">
            <strong>Exists:</strong> {data ? "✅ Yes" : "❌ No"}
          </div>
          {data && (
            <div className="mt-2">
              <strong>Type:</strong> {typeof data}
              {typeof data === "object" && (
                <>
                  <br />
                  <strong>Keys:</strong> {Object.keys(data).join(", ")}
                  <br />
                  <details className="mt-2">
                    <summary className="cursor-pointer text-blue-600">
                      View Raw Data
                    </summary>
                    <pre className="mt-2 p-2 bg-gray-100 rounded text-xs overflow-auto max-h-40">
                      {JSON.stringify(data, null, 2)}
                    </pre>
                  </details>
                </>
              )}
            </div>
          )}
        </div>
      ))}

      <div className="mt-6 p-4 bg-blue-50 rounded border border-blue-200">
        <h3 className="font-semibold text-blue-800 mb-2">
          Quick Fix Suggestions:
        </h3>
        <ul className="text-sm text-blue-700 space-y-1">
          <li>
            • If no SHAP data exists, check your backend simulation process
          </li>
          <li>
            • If data exists in different location, update SHAPDashboard
            component
          </li>
          <li>• Check browser console for any API errors</li>
          <li>• Verify database contains SHAP data for this simulation</li>
        </ul>
      </div>
    </div>
  );
};

export default SHAPDebugger;
