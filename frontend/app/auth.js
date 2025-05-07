// app/auth.js
"use client";
import { useState } from "react";
import axios from "axios";

export default function AuthPage() {
  const [isLogin, setIsLogin] = useState(true);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");


  const handleSubmit = async () => {
    setError("");
    setMessage("");

    try {
      // Ensure code runs only in browser (window & localStorage)
      if (typeof window === 'undefined') return;

      const origin = window.location.origin;

      if (isLogin) {
        // Login
        const response = await axios.post(
          `${origin}/auth/jwt/login`,
          new URLSearchParams({ username: email, password })
        );

        localStorage.setItem("jwt", response.data.access_token);
        setMessage("🎉 Login successful!");
        window.location.href = "/";
      } else {
        // Register
        await axios.post(`${origin}/auth/register`, {
          email,
          password,
          is_active: true,
          is_superuser: false,
          is_verified: false,
        });

        setMessage("✅ Registration successful! You can now log in.");
        setIsLogin(true);
      }

    } catch (err) {
      const errorMessage = err?.response?.data?.detail || "Something went wrong!";
      setError(errorMessage);
    }
  };


  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 to-gray-100 flex items-center justify-center">
      <div className="bg-white shadow-lg p-8 rounded-xl max-w-md w-full">
        <h2 className="text-2xl font-bold text-center mb-4">
          {isLogin ? "Login 🌿" : "Register 🌱"}
        </h2>

        <input
          type="email"
          placeholder="Email"
          className="border border-gray-300 rounded p-3 w-full mb-4"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
        />

        <input
          type="password"
          placeholder="Password"
          className="border border-gray-300 rounded p-3 w-full mb-4"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />

        {error && <div className="text-red-500 text-center mb-4">{error}</div>}
        {message && (
          <div className="text-green-600 text-center mb-4">{message}</div>
        )}

        <button
          className="bg-green-500 text-white rounded px-6 py-3 w-full"
          onClick={handleSubmit}
        >
          {isLogin ? "Log In" : "Register"}
        </button>

        <button
          className="text-sm text-gray-500 mt-4 w-full"
          onClick={() => setIsLogin(!isLogin)}
        >
          {isLogin
            ? "No account? Click here to register"
            : "Have an account? Click here to login"}
        </button>
      </div>
    </div>
  );
}
