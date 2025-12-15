import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import Nav from "./components/Nav";

export default function DisclaimerPage() {
  const navigate = useNavigate();
  const [checked, setChecked] = useState(false);
  const [isAuthenticated, setIsAuthenticated] = useState(null);

  useEffect(() => {
    const checkLoginStatus = async () => {
      try {
        const response = await fetch("/api/home", { credentials: "include" });
        if (response.ok) {
          setIsAuthenticated(true);
        } else {
          setIsAuthenticated(false);
          navigate("/login");
        }
      } catch {
        setIsAuthenticated(false);
        navigate("/login");
      }
    };
    checkLoginStatus();
  }, [navigate]);

  // // If already accepted, go straight to upload
  // useEffect(() => {
  //   const accepted = localStorage.getItem("dbcompare_upload_disclaimer_accepted");
  //   if (accepted === "true") navigate("/upload");
  // }, [navigate]);

  if (isAuthenticated === null) return null;

  const handleAgree = () => {
    sessionStorage.setItem("dbcompare_upload_disclaimer_pass", "true");
    navigate("/upload");
  };
  

  return (
    <>
      <Nav />
      <div className="min-h-screen bg-gray-100 flex items-start justify-center p-6">
        <div className="w-full max-w-2xl bg-white rounded-lg shadow-lg p-6 mt-10">
          <h1 className="text-2xl font-bold mb-2 text-gray-900">Before you upload</h1>
          <p className="text-gray-700 mb-4">
            DBCompare is a shared platform. Do <b>not</b> upload any personal, confidential, or sensitive information.
          </p>

          <div className="bg-yellow-50 border border-yellow-200 rounded p-4 text-gray-800">
            <ul className="list-disc pl-5 space-y-2">
              <li>No PII (names, addresses, phone numbers, emails, SSNs, student IDs, etc.).</li>
              <li>No health, financial, or private records.</li>
              <li>Use only synthetic/anonymized datasets.</li>
            </ul>
          </div>

          <label className="flex items-center gap-2 mt-5 text-gray-800">
            <input
              type="checkbox"
              checked={checked}
              onChange={(e) => setChecked(e.target.checked)}
            />
            I understand and agree not to upload personal/sensitive information.
          </label>

          <div className="flex gap-3 mt-6">
            <button
              onClick={() => navigate("/")}
              className="px-4 py-2 rounded bg-gray-200 hover:bg-gray-300"
            >
              Cancel
            </button>

            <button
              onClick={handleAgree}
              disabled={!checked}
              className={`px-4 py-2 rounded text-white ${
                checked ? "bg-blue-600 hover:bg-blue-700" : "bg-blue-300 cursor-not-allowed"
              }`}
            >
              I Agree & Proceed to Upload
            </button>
          </div>
        </div>
      </div>
    </>
  );
}
