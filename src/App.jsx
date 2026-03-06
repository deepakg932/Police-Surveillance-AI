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
import { workflowSteps, reportsData } from "./data/sampleData";

function Dashboard() {
  const [activeTab, setActiveTab] = useState("upload");
  const [uploadedContent, setUploadedContent] = useState(null);
  const [detectionData, setDetectionData] = useState({
    persons: [],
    stats: {
      totalPersons: 0,
      totalMotorcycles: 0,
      withHelmet: 0,
      processingTime: '0 ms'
    },
    loading: false,
    error: null
  });
  const { uploadProgress, isUploading, handleFileUpload } = useFileUpload();
  const { logout, user, authFetch } = useAuth();

  // Fetch detection data when tab changes to detection or after upload
  useEffect(() => {
    if (activeTab === "detection") {
      fetchDetectionData();
    }
  }, [activeTab]);

  const fetchDetectionData = async () => {
    setDetectionData(prev => ({ ...prev, loading: true, error: null }));
    
    try {
      // You can add query parameters if needed
      const response = await authFetch("http://192.168.29.248:5000/api/video/search");
      
      if (!response.ok) {
        throw new Error(`Failed to fetch data: ${response.status}`);
      }
      
      const data = await response.json();
      console.log("Detection data received:", data);
      
      // Transform API data to match your component structure
      const transformedData = transformApiData(data);
      
      setDetectionData({
        persons: transformedData.persons,
        stats: transformedData.stats,
        loading: false,
        error: null
      });
    } catch (error) {
      console.error("Error fetching detection data:", error);
      setDetectionData(prev => ({
        ...prev,
        loading: false,
        error: error.message
      }));
    }
  };

// Updated transformApiData function in your Dashboard component
const transformApiData = (apiData) => {
  console.log("Transforming API data:", apiData);
  
  // Check if apiData has the structure from your response
  if (apiData.results && Array.isArray(apiData.results)) {
    // Group detections by trackingId to create unique persons
    const personsMap = new Map();
    
    apiData.results.forEach((item, index) => {
      const trackingId = item.trackingId || `unknown_${index}`;
      
      if (!personsMap.has(trackingId)) {
        personsMap.set(trackingId, {
          id: trackingId,
          trackingId: trackingId,
          startTime: item.timestamp || '00:00',
          endTime: item.timestamp || '00:00', // You might want to calculate this properly
          blueShirt: false, // Not in your response
          helmet: true, // Since all are "person with helmet"
          motorcycle: false, // Not in your response
          confidence: Math.round(item.confidence * 100), // Convert to percentage
          thumbnail: item.screenshotUrl || item.image_path,
          timestamp: item.timestamp,
          image_path: item.image_path,
          screenshotUrl: item.screenshotUrl
        });
      } else {
        // Update end time if this is a later detection of the same person
        const existingPerson = personsMap.get(trackingId);
        if (parseInt(item.timestamp) > parseInt(existingPerson.endTime)) {
          existingPerson.endTime = item.timestamp;
        }
      }
    });

    const persons = Array.from(personsMap.values());
    
    // Calculate stats based on your response
    const stats = {
      totalPersons: apiData.totalUniqueObjects || persons.length,
      totalMotorcycles: 0, // Not in your response
      withHelmet: apiData.counts?.["person with helmet"] || persons.length,
      processingTime: 'N/A' // Not in your response
    };

    return { persons, stats };
  }
  
  // Handle other possible response structures
  if (apiData.persons && apiData.stats) {
    return {
      persons: apiData.persons.map(person => ({
        id: person.id || person.trackingId,
        startTime: person.startTime || person.start_time || person.timestamp,
        endTime: person.endTime || person.end_time || person.timestamp,
        blueShirt: person.blueShirt || person.blue_shirt || false,
        helmet: person.helmet || person.object?.includes('helmet') || false,
        motorcycle: person.motorcycle || false,
        confidence: person.confidence ? Math.round(person.confidence * 100) : person.confidence,
        thumbnail: person.thumbnail || person.screenshotUrl || person.image_path
      })),
      stats: {
        totalPersons: apiData.stats.totalPersons || apiData.totalUniqueObjects || apiData.persons.length,
        totalMotorcycles: apiData.stats.totalMotorcycles || 0,
        withHelmet: apiData.stats.withHelmet || apiData.counts?.["person with helmet"] || 0,
        processingTime: apiData.stats.processingTime || 'N/A'
      }
    };
  }

  // Default structure
  return {
    persons: [],
    stats: {
      totalPersons: 0,
      totalMotorcycles: 0,
      withHelmet: 0,
      processingTime: 'N/A'
    }
  };
};

  const handleSubmit = async (data) => {
    console.log("Submitting:", data);
    setUploadedContent(data);

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

      // if (user?.id) {
      //   formData.append("userId", user.id);
      // }

      for (let pair of formData.entries()) {
        console.log(
          pair[0] + ": " + (pair[1] instanceof File ? pair[1].name : pair[1]),
        );
      }

      const response = await authFetch(
        "http://192.168.29.248:5000/api/video/upload",
        {
          method: "POST",
          body: formData,
        },
      );

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
        `http://192.168.29.248:5000/api/search?${queryParams}`,
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
          error: null
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
                <DetectionStats stats={detectionData.stats} />
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