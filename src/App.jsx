import React, { useState, useEffect } from "react";
import {
  BrowserRouter as Router,
  Routes,
  Route,
  Navigate,
  useNavigate,
} from "react-router-dom";
import { AuthProvider, useAuth } from "./components/contexts/AuthContext";
import Header from "./components/Layout/Header";
import Navigation from "./components/Layout/Navigation";
import VideoUploadForm from "./components/Upload/VideoUploadForm";
import DetectionStats from "./components/Detection/DetectionStats";
import WorkflowProgress from "./components/Detection/WorkflowProgress";
import DetectedPersonsTable from "./components/Detection/DetectedPersonsTable";
import SearchForm from "./components/Search/SearchForm";
import ReportsList from "./components/Reports/ReportsList";
import LoginForm from "./components/Auth/LoginForm";
import RegisterForm from "./components/Auth/RegisterForm";
import ProtectedRoute from "./components/Auth/ProtectedRoute";
import PublicRoute from "./components/Auth/PublicRoute";
import { useFileUpload } from "./hooks/useFileUpload";
import { reportsData } from "./data/sampleData";

function Dashboard() {
  const [activeTab, setActiveTab] = useState("upload");
  const API_BASE_URL = import.meta.env.VITE_API_BASE_URL;
  const [detectionData, setDetectionData] = useState({
    persons: [],
    stats: {
      totalPersons: 0,
      withHelmet: 0,
      processingTime: "N/A",
    },
    loading: false,
    error: null,
    mode: null,
  });

  console.log(detectionData, "fffffffffffffffffffffffffffffffff");
  const { uploadProgress, isUploading } = useFileUpload();
  const { logout, user, authFetch } = useAuth();

  // Fetch detection data when tab changes to detection or after upload
  useEffect(() => {
    if (activeTab === "detection") {
      fetchDetectionData();
    }
  }, [activeTab]);

  const fetchDetectionData = async () => {
    setDetectionData((prev) => ({ ...prev, loading: true, error: null }));

    try {
      const response = await authFetch(`${API_BASE_URL}/video/search`);

      if (!response.ok) {
        throw new Error(`Failed to fetch data: ${response.status}`);
      }

      const data = await response.json();
      console.log("Detection data received:", data);

      // Transform API data to match your component structure
      const transformedData = transformApiData(data);

      console.log("Transformed data:", transformedData);

      setDetectionData({
        persons: transformedData.persons,
        stats: transformedData.stats,
        loading: false,
        error: null,
        mode: data.mode,
      });
    } catch (error) {
      console.error("Error fetching detection data:", error);
      setDetectionData((prev) => ({
        ...prev,
        loading: false,
        error: error.message,
      }));
    }
  };

  const transformApiData = (apiData) => {
    console.log("Raw API data:", apiData);

    // Handle the response structure with results array
    if (apiData.results && Array.isArray(apiData.results)) {
      // Process results into persons array
      const persons = apiData.results.map((item, index) => {
        // Get the object type from the object field
        const objectType = item.object || "unknown";

        // Determine if it's a helmet based on object field
        const hasHelmet = objectType.toLowerCase().includes("helmet");

        // For other attributes, since the object is just "helmet" in your response,
        // we set appropriate defaults
        const hasBlack = false; // Not provided in response
        const hasShirt = false; // Not provided in response
        const hasPerson = objectType.toLowerCase().includes("person");

        // Get the image URL - prioritize screenshotUrl, then image_path
        let imageUrl = null;
        if (item.screenshotUrl) {
          imageUrl = item.screenshotUrl;
        } else if (item.image_path) {
          // Check if it's a full URL or relative path
          if (item.image_path.startsWith("http")) {
            imageUrl = item.image_path;
          } else {
            // If it's a relative path, construct the full URL
            // Use the base URL from your API or environment
            const baseUrl = API_BASE_URL || "http://localhost:5000";
            imageUrl = `${baseUrl}/${item.image_path}`;
          }
        }

        return {
          id: item.trackingId || `unknown_${index}`,
          trackingId: item.trackingId,
          object: item.object,
          startTime: item.timestamp || "00:00",
          endTime: item.timestamp || "00:00",
          // Parse attributes from object string
          helmet: hasHelmet,
          blackShirt: hasBlack,
          shirt: hasShirt,
          person: hasPerson,
          confidence: item.confidence ? Math.round(item.confidence * 100) : 0,
          thumbnail: imageUrl,
          image_path: item.image_path,
          screenshotUrl: item.screenshotUrl,
          bbox: item.bbox,
          timestamp: item.timestamp,
          fullImageUrl: imageUrl,
          processing_time: item.processing_time,
        };
      });

      // Create dynamic stats from counts object
      const stats = {
        totalPersons: apiData.totalUniqueObjects || persons.length,
        processingTime: apiData.results[0]?.processing_time || "N/A",
      };

      // Add all counts from the counts object dynamically
      if (apiData.counts) {
        Object.entries(apiData.counts).forEach(([key, value]) => {
          // Convert count keys to readable labels
          let label = key;
          if (key === "total_helmet") {
            label = "withHelmet";
          } else if (key === "total_person") {
            label = "totalPersons";
          } else {
            // Remove 'total_' prefix and convert to camelCase
            label = key
              .replace(/^total_/, "")
              .replace(/_([a-z])/g, (_, letter) => letter.toUpperCase());
          }
          stats[label] = value;
        });
      }

      // Ensure withHelmet is set from total_helmet if available
      if (apiData.counts?.total_helmet !== undefined) {
        stats.withHelmet = apiData.counts.total_helmet;
      } else {
        // If no helmet count, calculate from persons array
        stats.withHelmet = persons.filter((p) => p.helmet).length;
      }

      console.log("Transformed persons:", persons);
      console.log("Transformed stats:", stats);

      return { persons, stats };
    }

    // Handle previous response structure with persons and stats
    if (apiData.persons && apiData.stats) {
      return {
        persons: apiData.persons.map((person) => {
          // Get the image URL
          let imageUrl = null;
          if (person.screenshotUrl) {
            imageUrl = person.screenshotUrl;
          } else if (person.image_path) {
            if (person.image_path.startsWith("http")) {
              imageUrl = person.image_path;
            } else {
              const baseUrl = API_BASE_URL || "http://localhost:5000";
              imageUrl = `${baseUrl}/${person.image_path}`;
            }
          } else if (person.thumbnail) {
            imageUrl = person.thumbnail;
          }

          return {
            id: person.id || person.trackingId,
            startTime:
              person.startTime || person.start_time || person.timestamp,
            endTime: person.endTime || person.end_time || person.timestamp,
            blueShirt: person.blueShirt || person.blue_shirt || false,
            helmet: person.helmet || person.object?.includes("helmet") || false,
            motorcycle: person.motorcycle || false,
            confidence: person.confidence
              ? Math.round(person.confidence * 100)
              : person.confidence,
            thumbnail: imageUrl,
            image_path: person.image_path,
            screenshotUrl: person.screenshotUrl,
            fullImageUrl: imageUrl,
            processing_time: person.processing_time,
          };
        }),
        stats: {
          totalPersons:
            apiData.stats.totalPersons ||
            apiData.totalUniqueObjects ||
            apiData.persons.length,
          totalMotorcycles: apiData.stats.totalMotorcycles || 0,
          withHelmet:
            apiData.stats.withHelmet || apiData.counts?.total_helmet || 0,
          processingTime: apiData.persons.processing_time || "N/A",
        },
      };
    }

    // Default structure
    return {
      persons: [],
      stats: {
        totalPersons: 0,
        withHelmet: 0,
        processingTime: "N/A",
      },
    };
  };

  const handleSubmit = async (data) => {
    console.log("Submitting:", data);

    try {
      const videoFile = data.file || data.video;

      if (!videoFile) {
        throw new Error("No video file selected");
      }

      const formData = new FormData();
      formData.append("file", videoFile, videoFile.name);

      if (data.image) {
        formData.append("image", data.image, data.image.name);
      }

      if (data.text) {
        formData.append("text", data.text);
      }

      for (let pair of formData.entries()) {
        console.log(
          pair[0] + ": " + (pair[1] instanceof File ? pair[1].name : pair[1]),
        );
      }

      const response = await authFetch(`${API_BASE_URL}/video/upload`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Upload failed: ${response.status} - ${errorText}`);
      }

      const result = await response.json();
      console.log("Upload successful:", result);

      alert("Video uploaded successfully!");

      // Fetch the latest detection data after successful upload
      await fetchDetectionData();
      setActiveTab("detection");
    } catch (error) {
      console.error("Upload failed:", error);
    }
  };

  const handleSearch = async (searchData) => {
    console.log("Searching with:", searchData);
    try {
      const queryParams = new URLSearchParams(searchData).toString();
      const response = await authFetch(
        `${API_BASE_URL}/video/search?${queryParams}`,
      );

      if (response.ok) {
        const results = await response.json();
        console.log("Search results:", results);

        // Transform and update the detection data with search results
        const transformedData = transformApiData(results);
        setDetectionData({
          persons: transformedData.persons,
          stats: transformedData.stats,
          loading: false,
          error: null,
          mode: results.mode,
        });
      }
    } catch (error) {
      console.error("Search failed:", error);
    }
  };

  const renderContent = () => {
    switch (activeTab) {
      case "upload":
        return (
          <VideoUploadForm
            isUploading={isUploading}
            uploadProgress={uploadProgress}
            onFileUpload={() => {}}
            onSubmit={handleSubmit}
          />
        );

      case "detection":
        return (
          <div className="space-y-6">
            {detectionData.loading ? (
              <div className="text-center py-12">
                <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
                <p className="text-gray-400 mt-4">Loading detection data...</p>
              </div>
            ) : detectionData.error ? (
              <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-6 text-center">
                <p className="text-red-400">Error: {detectionData.error}</p>
                <button
                  onClick={fetchDetectionData}
                  className="mt-4 px-4 py-2 bg-red-500/20 text-red-400 rounded-lg hover:bg-red-500/30"
                >
                  Retry
                </button>
              </div>
            ) : (
              <>
                <DetectionStats
                  stats={detectionData.stats}
                  mode={detectionData.mode}
                />
                <DetectedPersonsTable persons={detectionData.persons} />
              </>
            )}
          </div>
        );

      case "search":
        return <SearchForm onSearch={handleSearch} />;

      case "reports":
        return <ReportsList reports={reportsData} />;

      default:
        return null;
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900">
      <Header user={user} onLogout={logout} />
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <Navigation activeTab={activeTab} onTabChange={setActiveTab} />
        {renderContent()}
      </main>
    </div>
  );
}

function LoginPage() {
  const { login, isLoading, error } = useAuth();
  const navigate = useNavigate();

  const handleLogin = async (credentials) => {
    const success = await login(credentials);
    if (success) {
      navigate("/");
    }
  };

  const handleSwitchToRegister = () => {
    navigate("/register");
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 flex items-center justify-center p-4">
      <LoginForm
        onLogin={handleLogin}
        isLoading={isLoading}
        error={error}
        onSwitchToRegister={handleSwitchToRegister}
      />
    </div>
  );
}

function RegisterPage() {
  const { register, isLoading, error } = useAuth();
  const navigate = useNavigate();

  const handleRegister = async (userData) => {
    const success = await register(userData);
    if (success) {
      navigate("/");
    }
  };

  const handleSwitchToLogin = () => {
    navigate("/login");
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 flex items-center justify-center p-4">
      <RegisterForm
        onRegister={handleRegister}
        isLoading={isLoading}
        error={error}
        onSwitchToLogin={handleSwitchToLogin}
      />
    </div>
  );
}

function App() {
  return (
    <Router>
      <AuthProvider>
        <Routes>
          <Route
            path="/login"
            element={
              <PublicRoute>
                <LoginPage />
              </PublicRoute>
            }
          />
          <Route
            path="/register"
            element={
              <PublicRoute>
                <RegisterPage />
              </PublicRoute>
            }
          />
          <Route
            path="/"
            element={
              <ProtectedRoute>
                <Dashboard />
              </ProtectedRoute>
            }
          />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </AuthProvider>
    </Router>
  );
}

export default App;
