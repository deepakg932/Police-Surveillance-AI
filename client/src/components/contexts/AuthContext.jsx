import React, {
  createContext,
  useState,
  useContext,
  useEffect,
  useCallback,
  useRef,
} from "react";
// import VITE_API_BASE_URL from "";

const AuthContext = createContext(null);

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
};

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  // Use a ref to prevent multiple logout calls
  const isLoggingOut = useRef(false);

  // Base API URL
  const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://192.168.29.248:5000/api-auth";

  // Validate token with server
  // const validateToken = async (token) => {
  //   try {
  //     const response = await fetch(`${API_BASE_URL}/validate`, {
  //       method: "GET",
  //       headers: {
  //         Authorization: `Bearer ${token}`,
  //         "Content-Type": "application/json",
  //       },
  //     });
  //     return response.ok;
  //   } catch (error) {
  //     console.error("Token validation failed:", error);
  //     return false;
  //   }
  // };

  // Clear all auth data
  const clearAuthData = useCallback(() => {
    console.log("Clearing auth data...");

    // Clear state
    setUser(null);
    setToken(null);

    // Clear all storage
    localStorage.removeItem("user");
    localStorage.removeItem("token");
    localStorage.removeItem("rememberMe");
    sessionStorage.removeItem("user");
    sessionStorage.removeItem("token");

    // Clear any other auth-related items
    localStorage.removeItem("refreshToken");
    sessionStorage.removeItem("refreshToken");

    console.log("Auth data cleared");
  }, []);

  // Check for saved session on mount
  // Check for saved session on mount
  useEffect(() => {
    const checkSavedSession = async () => {
      try {
        // Check localStorage first (remember me)
        let savedUser = localStorage.getItem("user");
        let savedToken = localStorage.getItem("token");

        // If not in localStorage, check sessionStorage
        if (!savedUser || !savedToken) {
          savedUser = sessionStorage.getItem("user");
          savedToken = sessionStorage.getItem("token");
        }

        if (savedUser && savedToken) {
          console.log("Found saved session, restoring...");

          // Parse user data and restore session
          const parsedUser = JSON.parse(savedUser);
          setUser(parsedUser);
          setToken(savedToken);

          console.log("Session restored successfully");
        }
      } catch (error) {
        console.error("Session restoration failed:", error);
        // Clear invalid data if any
        localStorage.removeItem("user");
        localStorage.removeItem("token");
        sessionStorage.removeItem("user");
        sessionStorage.removeItem("token");
      } finally {
        setIsLoading(false);
      }
    };

    checkSavedSession();
  }, []); // Empty dependency array is fine

  const register = async (userData) => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE_URL}-auth/register`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          email: userData.email,
          password: userData.password,
          full_name: userData.fullName,
          badge_number: userData.badgeNumber,
          department: userData.department,
          phone: userData.phone,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.message || data.error || "Registration failed");
      }

      // If API returns user and token after registration
      if (data.user && data.token) {
        setUser(data.user);
        setToken(data.token);

        // Save to storage
        localStorage.setItem("user", JSON.stringify(data.user));
        localStorage.setItem("token", data.token);
        localStorage.setItem("rememberMe", "true");
      }

      return true;
    } catch (err) {
      setError(err.message || "An error occurred during registration");
      return false;
    } finally {
      setIsLoading(false);
    }
  };

  const login = async (credentials) => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE_URL}-auth/login`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          email: credentials.email,
          password: credentials.password,
        }),
      });

      const data = await response.json();

      console.log("Login response:", data);

      if (!response.ok) {
        throw new Error(data.message || data.error || "Login failed");
      }

      // Handle different API response structures
      const userData = data.user || data.data?.user || data;
      const authToken = data.token || data.user?.token || data.access_token;

      if (!userData || !authToken) {
        throw new Error("Invalid response format from server");
      }

      setUser(userData);
      setToken(authToken);

      // Save to appropriate storage based on remember me
      if (credentials.rememberMe) {
        localStorage.setItem("user", JSON.stringify(userData));
        localStorage.setItem("token", authToken);
        localStorage.setItem("rememberMe", "true");
      } else {
        sessionStorage.setItem("user", JSON.stringify(userData));
        sessionStorage.setItem("token", authToken);
        localStorage.removeItem("rememberMe"); // Clear remember me flag
      }

      return true;
    } catch (err) {
      setError(err.message || "An error occurred during login");
      return false;
    } finally {
      setIsLoading(false);
    }
  };

  const logout = useCallback(async () => {
    // Prevent multiple simultaneous logout calls
    if (isLoggingOut.current) {
      console.log("Logout already in progress...");
      return;
    }

    isLoggingOut.current = true;
    console.log("Starting logout process...");

    // Clear all auth data
    clearAuthData();

    // Reset logout flag after a short delay
    setTimeout(() => {
      isLoggingOut.current = false;
    }, 1000);

    console.log("Logout completed");
  }, [clearAuthData]);

  const hasPermission = (requiredRole) => {
    if (!user) return false;

    const roleHierarchy = {
      admin: ["admin", "supervisor", "officer"],
      supervisor: ["supervisor", "officer"],
      officer: ["officer"],
    };

    return roleHierarchy[user.role]?.includes(requiredRole) || false;
  };

  // Auth fetch wrapper for API calls
  const authFetch = useCallback(
    async (url, options = {}) => {
      // Don't include Content-Type for FormData
      const isFormData = options.body instanceof FormData;

      const headers = {
        ...(isFormData ? {} : { "Content-Type": "application/json" }),
        ...options.headers,
      };

      if (token) {
        headers["Authorization"] = `Bearer ${token}`;
      }

      try {
        const response = await fetch(url, {
          ...options,
          headers,
        });

        if (response.status === 401) {
          console.log("Token expired, logging out...");
          // Token expired - logout
          await logout();
          throw new Error("Session expired. Please login again.");
        }

        return response;
      } catch (error) {
        // Handle network errors
        if (error.name === "TypeError" && error.message === "Failed to fetch") {
          throw new Error(
            "Network error. Please check your internet connection.",
          );
        }

        // Re-throw the error
        throw error;
      }
    },
    [token, logout],
  );

  const value = {
    user,
    token,
    isLoading,
    error,
    login,
    register,
    logout,
    hasPermission,
    authFetch,
    isAuthenticated: !!user && !!token,
    isAdmin: user?.role === "admin",
    isSupervisor: user?.role === "supervisor",
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};
