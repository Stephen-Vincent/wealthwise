import React, { useState, useEffect, useRef } from "react";
import { ChevronDown, Check } from "lucide-react";

// Custom Dropdown Component
const CustomDropdown = ({
  options,
  placeholder,
  value,
  onChange,
  className = "",
  maxHeight = "200px",
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [searchTerm, setSearchTerm] = useState("");
  const dropdownRef = useRef(null);
  const inputRef = useRef(null);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setIsOpen(false);
        setSearchTerm("");
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  // Filter options based on search term
  const filteredOptions = options.filter((option) =>
    option.label.toLowerCase().includes(searchTerm.toLowerCase())
  );

  // Get selected option for display
  const selectedOption = options.find((option) => option.value === value);

  const handleOptionClick = (option) => {
    onChange(option.value);
    setIsOpen(false);
    setSearchTerm("");
  };

  const handleInputChange = (e) => {
    setSearchTerm(e.target.value);
    if (!isOpen) setIsOpen(true);
  };

  const handleToggle = () => {
    setIsOpen(!isOpen);
  };

  return (
    <>
      {/* Backdrop overlay when dropdown is open */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-transparent z-[999]"
          onClick={() => setIsOpen(false)}
        />
      )}

      <div
        ref={dropdownRef}
        className={`relative ${className}`}
        style={{ zIndex: isOpen ? 1000 : 1 }}
      >
        <div
          className="relative w-full h-[70px] border-2 border-[#00A8FF] rounded-[15px] px-4 pr-12 bg-white cursor-pointer transition-all duration-300 hover:border-[#0088CC] focus-within:border-[#0088CC] focus-within:ring-2 focus-within:ring-[#00A8FF]/20"
          onClick={handleToggle}
        >
          <input
            ref={inputRef}
            type="text"
            className="w-full h-full text-lg font-bold bg-transparent outline-none cursor-pointer"
            placeholder={placeholder}
            value={isOpen ? searchTerm : selectedOption?.label || ""}
            onChange={handleInputChange}
            onFocus={() => setIsOpen(true)}
            readOnly={!isOpen}
          />
          <div className="absolute inset-y-0 right-0 flex items-center pr-4 pointer-events-none">
            <ChevronDown
              className={`w-6 h-6 text-[#00A8FF] transition-transform duration-200 ${
                isOpen ? "rotate-180" : ""
              }`}
            />
          </div>
        </div>

        {isOpen && (
          <div
            className="absolute w-full mt-2 bg-white border-2 border-[#00A8FF] rounded-[15px] shadow-2xl overflow-hidden"
            style={{
              zIndex: 1001,
              boxShadow:
                "0 25px 50px -12px rgba(0, 0, 0, 0.25), 0 0 0 1px rgba(0, 168, 255, 0.1)",
            }}
          >
            <div
              className="max-h-[200px] overflow-y-auto scrollbar-thin scrollbar-thumb-[#00A8FF] scrollbar-track-gray-100"
              style={{ maxHeight }}
            >
              {filteredOptions.length > 0 ? (
                filteredOptions.map((option) => (
                  <div
                    key={option.value}
                    className="px-4 py-3 text-lg font-medium hover:bg-[#00A8FF] hover:text-white cursor-pointer transition-colors duration-200 flex items-center justify-between"
                    onClick={() => handleOptionClick(option)}
                  >
                    <span>{option.label}</span>
                    {value === option.value && (
                      <Check className="w-5 h-5 text-[#00A8FF] hover:text-white" />
                    )}
                  </div>
                ))
              ) : (
                <div className="px-4 py-3 text-lg text-gray-500 text-center">
                  No options found
                </div>
              )}
            </div>
          </div>
        )}

        <style>{`
          .scrollbar-thin {
            scrollbar-width: thin;
          }

          .scrollbar-thumb-blue-500 {
            scrollbar-color: #00A8FF #f1f1f1;
          }

          .scrollbar-track-gray-100 {
            scrollbar-color: #00A8FF #f1f1f1;
          }

          /* Custom scrollbar for webkit browsers */
          .scrollbar-thin::-webkit-scrollbar {
            width: 6px;
          }

          .scrollbar-thin::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 10px;
          }

          .scrollbar-thin::-webkit-scrollbar-thumb {
            background: #00A8FF;
            border-radius: 10px;
          }

          .scrollbar-thin::-webkit-scrollbar-thumb:hover {
            background: #0088CC;
          }
        `}</style>
      </div>
    </>
  );
};

export default CustomDropdown;
