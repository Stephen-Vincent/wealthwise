/**
 * This file contains three progress indicator components for tracking user progress in multi-step flows:
 * - ProgressDots: displays a series of dots indicating the current step.
 * - ProgressBar: shows a progress bar with percentage and step count.
 * - NumberedProgressSteps: renders numbered steps with visual indication of progress.
 */
import React from "react";

// Progress Dots Component
const ProgressDots = ({ total, current, className = "" }) => (
  <div className={`flex space-x-2 mt-6 ${className}`}>
    {Array.from({ length: total }, (_, i) => (
      <div
        key={i}
        className={`w-3 h-3 rounded-full transition-colors duration-300 ${
          i <= current ? "bg-[#00A8FF]" : "bg-gray-300"
        }`}
      />
    ))}
  </div>
);

// Alternative Progress Bar Component
export const ProgressBar = ({ total, current, className = "" }) => {
  const percentage = Math.round((current / (total - 1)) * 100);

  return (
    <div className={`w-full mt-6 ${className}`}>
      <div className="flex justify-between text-sm text-gray-600 mb-2">
        <span>
          Step {current + 1} of {total}
        </span>
        <span>{percentage}%</span>
      </div>
      <div className="w-full bg-gray-200 rounded-full h-2">
        <div
          className="bg-[#00A8FF] h-2 rounded-full transition-all duration-500 ease-out"
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
};

// Numbered Progress Steps Component
export const NumberedProgressSteps = ({ total, current, className = "" }) => (
  <div
    className={`flex items-center justify-center space-x-4 mt-6 ${className}`}
  >
    {Array.from({ length: total }, (_, i) => (
      <React.Fragment key={i}>
        <div
          className={`w-10 h-10 rounded-full flex items-center justify-center text-sm font-bold transition-all duration-300 ${
            i <= current
              ? "bg-[#00A8FF] text-white"
              : "bg-gray-300 text-gray-600"
          }`}
        >
          {i + 1}
        </div>
        {i < total - 1 && (
          <div
            className={`w-8 h-1 transition-colors duration-300 ${
              i < current ? "bg-[#00A8FF]" : "bg-gray-300"
            }`}
          />
        )}
      </React.Fragment>
    ))}
  </div>
);

export default ProgressDots;
