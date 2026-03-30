import React, { useState, useEffect } from "react";
import {
  BrowserRouter as Router,
  Routes,
  Route,
  Navigate,
  useNavigate,
} from "react-router-dom";
import { Toaster, toast } from "react-hot-toast";
import { AuthProvider, useAuth } from "./components/contexts/AuthContext";
import { UploadProvider } from "./components/contexts/UploadContext";
import {
  HistoryProvider,
  useHistory,
} from "./components/contexts/HistoryContext";
import Header from "./components/Layout/Header";
import Navigation from "./components/Layout/Navigation";
import UploadSection from "./components/Upload/UploadSection";
import DetectionStats from "./components/Detection/DetectionStats";
import DetectedPersonsTable from "./components/Detection/DetectedPersonsTable";
import PreviousResultsTable from "./components/Detection/PreviousResultsTable";
import SearchForm from "./components/Search/SearchForm";
import ReportsList from "./components/Reports/ReportsList";
import LoginForm from "./components/Auth/LoginForm";
import RegisterForm from "./components/Auth/RegisterForm";
import ProtectedRoute from "./components/Auth/ProtectedRoute";
import PublicRoute from "./components/Auth/PublicRoute";
import { reportsData } from "./data/sampleData";
import { useUpload } from "./components/contexts/UploadContext";

function Dashboard() {
  const [activeTab, setActiveTab] = useState("upload");
  const API_BASE_URL = import.meta.env.VITE_API_BASE_URL;
  const [detectionData, setDetectionData] = useState({
    persons: [],
    stats: {
      totalPersons: 0,
      totalHelmet: 0,
      totalVehicles: 0,
      totalDetections: 0,
      processingTime: "N/A",
      filters: {},
    },
    loading: false,
    error: null,
    mode: null,
  });
  const [viewingHistoryEntry, setViewingHistoryEntry] = useState(null);
  const [currentJobId, setCurrentJobId] = useState(null);
  const [jobPolling, setJobPolling] = useState(false);
  const [hasCurrentResults, setHasCurrentResults] = useState(false); // Track if we have results from current upload
  const { logout, user, authFetch } = useAuth();
  const { isUploading, uploadProgress, resetUpload } = useUpload();
  const { addToHistory, uploadHistory } = useHistory();

  // REMOVE this useEffect - we don't want to auto-fetch when switching to current tab
  // useEffect(() => {
  //   if (activeTab === "current") {
  //     fetchDetectionData();
  //   }
  // }, [activeTab]);

  const handleTabChange = (tabId) => {
    setActiveTab(tabId);
    // Clear history view when switching away
    if (tabId !== "history-view") {
      setViewingHistoryEntry(null);
    }
  };

  // Remove fetchDetectionData function entirely since we don't need it
  // Keep it commented or delete it

  const transformApiData = (apiData) => {
    console.log("Raw API data:", apiData);

    if (apiData.results && Array.isArray(apiData.results)) {
      const persons = apiData.results.map((item, index) => {
        const objectType = item.object || "unknown";

        const hasHelmet = objectType.toLowerCase().includes("helmet");
        const hasVehicle =
          objectType.toLowerCase().includes("vehicle") ||
          objectType.toLowerCase().includes("car") ||
          objectType.toLowerCase().includes("bike") ||
          objectType.toLowerCase().includes("motorcycle") ||
          objectType.toLowerCase().includes("scooter");
        const hasPerson = objectType.toLowerCase().includes("person");

        let imageUrl = null;
        if (item.screenshotUrl) {
          imageUrl = item.screenshotUrl;
        } else if (item.image_path) {
          if (item.image_path.startsWith("http")) {
            imageUrl = item.image_path;
          } else {
            const baseUrl = API_BASE_URL || "http://localhost:5000";
            imageUrl = `${baseUrl}/files/${item.image_path}`;
          }
        }

        const confidencePercentage = item.confidence
          ? Math.round(item.confidence * 100)
          : 0;

        return {
          id: item.trackingId || item.id || `detection_${index}`,
          trackingId: item.trackingId,
          object: item.object,
          startTime: item.timestamp || item.startTime || "00:00",
          endTime: item.timestamp || item.endTime || "00:00",
          helmet: hasHelmet,
          vehicle: hasVehicle,
          person: hasPerson,
          confidencePercentage: confidencePercentage,
          thumbnail: imageUrl,
          image_path: item.image_path,
          screenshotUrl: item.screenshotUrl,
          bbox: item.bbox,
          timestamp: item.timestamp,
          fullImageUrl: imageUrl,
          processing_time: item.processing_time,
          ...item,
        };
      });

      const stats = {
        totalPersons: apiData.totalPersons || 0,
        totalHelmet: apiData.totalHelmet || 0,
        totalVehicles: apiData.totalVehicles || 0,
        totalDetections: apiData.totalDetections || persons.length,
        processingTime:
          apiData.processing_time ||
          apiData.results[0]?.processing_time ||
          "N/A",
        filters: apiData.filters || {},
      };

      return { persons, stats };
    }

    return {
      persons: [],
      stats: {
        totalPersons: apiData.totalPersons || 0,
        totalHelmet: apiData.totalHelmet || 0,
        totalVehicles: apiData.totalVehicles || 0,
        totalDetections: apiData.totalDetections || 0,
        processingTime: apiData.processing_time || "N/A",
        filters: apiData.filters || {},
      },
    };
  };

  const pollJobStatus = async (jobId, videoFile, formInput) => {
    setJobPolling(true);
    setCurrentJobId(jobId);
    setHasCurrentResults(false); // Reset current results flag
    setDetectionData({
      persons: [],
      stats: {
        totalPersons: 0,
        totalHelmet: 0,
        totalVehicles: 0,
        totalDetections: 0,
        processingTime: "N/A",
        filters: {},
      },
      loading: true,
      error: null,
      mode: null,
    });

    const pollingToast = toast.loading("Video is processing...");

    const maxAttempts = 720; // 720 * 5 sec = 60 min
    let attempts = 0;

    const interval = setInterval(async () => {
      attempts += 1;

      try {
        const response = await authFetch(
          `${API_BASE_URL}/video/status/${jobId}`,
        );

        if (!response.ok) {
          throw new Error(`Status check failed: ${response.status}`);
        }

        const statusData = await response.json();
        console.log("Job status:", statusData);

        if (statusData.status === "completed") {
          clearInterval(interval);
          toast.dismiss(pollingToast);

          const transformedData = transformApiData({
            results: statusData.results || [],
            processing_time: statusData.processingTime || "N/A",
            mode: formInput.text ? "Text Search" : "Video Upload",
            totalDetections: statusData.totalResults || 0,
          });

          // Add to history
          addToHistory({
            videoName: videoFile.name,
            videoFile: videoFile,
            prompt: formInput.text,
            image: formInput.image,
            text: formInput.text,
            results: transformedData.persons,
            stats: transformedData.stats,
            processingTime: statusData.processingTime || "N/A",
            mode: formInput.text ? "Text Search" : "Video Upload",
            thumbnail:
              transformedData.persons[0]?.screenshotUrl ||
              transformedData.persons[0]?.thumbnail,
            timestamp: new Date().toISOString(),
          });

          // Set current detection data
          setDetectionData({
            persons: transformedData.persons,
            stats: transformedData.stats,
            loading: false,
            error: null,
            mode: formInput.text ? "Text Search" : "Video Upload",
          });

          setHasCurrentResults(true); // Mark that we have current results
          setJobPolling(false);
          setCurrentJobId(null);
          setActiveTab("current"); // Switch to current tab only after results are ready

          toast.success(
            `✅ Processing complete! Found ${transformedData.persons.length} detection${transformedData.persons.length !== 1 ? "s" : ""}.`,
          );
          return;
        }

        if (statusData.status === "failed") {
          clearInterval(interval);
          toast.dismiss(pollingToast);

          setDetectionData({
            persons: [],
            stats: {
              totalPersons: 0,
              totalHelmet: 0,
              totalVehicles: 0,
              totalDetections: 0,
              processingTime: "N/A",
              filters: {},
            },
            loading: false,
            error: statusData.errorMessage || "Processing failed",
            mode: null,
          });
          setJobPolling(false);
          setCurrentJobId(null);
          setHasCurrentResults(false);

          toast.error(
            `Processing failed: ${statusData.errorMessage || "Unknown error"}`,
          );
          return;
        }

        // Update loading state to show processing
        if (statusData.status === "processing") {
          setDetectionData((prev) => ({
            ...prev,
            loading: true,
            error: null,
          }));
        }

        if (attempts >= maxAttempts) {
          clearInterval(interval);
          toast.dismiss(pollingToast);

          setDetectionData({
            persons: [],
            stats: {
              totalPersons: 0,
              totalHelmet: 0,
              totalVehicles: 0,
              totalDetections: 0,
              processingTime: "N/A",
              filters: {},
            },
            loading: false,
            error: "Processing timeout",
            mode: null,
          });
          setJobPolling(false);
          setCurrentJobId(null);
          setHasCurrentResults(false);

          toast.error(
            "Processing is taking too long. Please check again later.",
          );
        }
      } catch (error) {
        console.error("Polling error:", error);
        clearInterval(interval);
        toast.dismiss(pollingToast);

        setDetectionData({
          persons: [],
          stats: {
            totalPersons: 0,
            totalHelmet: 0,
            totalVehicles: 0,
            totalDetections: 0,
            processingTime: "N/A",
            filters: {},
          },
          loading: false,
          error: error.message,
          mode: null,
        });
        setJobPolling(false);
        setCurrentJobId(null);
        setHasCurrentResults(false);

        toast.error(`Polling failed: ${error.message}`);
      }
    }, 5000);
  };

  const handleSubmit = async (data) => {
    console.log("Submitting:", data);

    let uploadingToast = null;

    try {
      const videoFile = data.file || data.video;

      if (!videoFile) {
        toast.error("No video file selected");
        return;
      }

      const formData = new FormData();
      formData.append("file", videoFile, videoFile.name);

      if (data.image) {
        formData.append("image", data.image, data.image.name);
      }

      if (data.text) {
        formData.append("text", data.text);
      }

      uploadingToast = toast.loading("Uploading video...");

      const response = await authFetch(`${API_BASE_URL}/video/upload`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Upload failed: ${response.status} - ${errorText}`);
      }

      const result = await response.json();
      console.log("Upload response:", result);

      toast.dismiss(uploadingToast);

      // RESET THE UPLOAD FORM IMMEDIATELY AFTER SUCCESSFUL UPLOAD
      resetUpload(); // This clears all form data in UploadContext

      if (result.jobId) {
        toast.success("Upload successful. Processing started.");
        // Switch to current tab immediately to show processing state
        setActiveTab("current");
        // Start polling with jobId
        pollJobStatus(result.jobId, videoFile, data);
        return;
      }

      // fallback: old sync backend still returning final results
      const transformedData = transformApiData(result);

      addToHistory({
        videoName: videoFile.name,
        videoFile: videoFile,
        prompt: data.text,
        image: data.image,
        text: data.text,
        results: transformedData.persons,
        stats: transformedData.stats,
        processingTime:
          result.processing_time || transformedData.stats.processingTime,
        mode: result.mode,
        thumbnail:
          transformedData.persons[0]?.screenshotUrl ||
          transformedData.persons[0]?.thumbnail,
        timestamp: new Date().toISOString(),
      });

      setDetectionData({
        persons: transformedData.persons,
        stats: transformedData.stats,
        loading: false,
        error: null,
        mode: result.mode,
      });

      setHasCurrentResults(true);
      setActiveTab("current");

      toast.success("Upload complete.");
    } catch (error) {
      console.error("Upload failed:", error);
      if (uploadingToast) toast.dismiss(uploadingToast);
      toast.error(`Upload failed: ${error.message}`);
    }
  };

  const handleSearch = async (searchData) => {
    console.log("Searching with:", searchData);

    const searchingToast = toast.loading("Searching...");

    try {
      const queryParams = new URLSearchParams(searchData).toString();
      const response = await authFetch(
        `${API_BASE_URL}/video/search?${queryParams}`,
      );

      if (response.ok) {
        const results = await response.json();
        console.log("Search results:", results);

        const transformedData = transformApiData(results);
        setDetectionData({
          persons: transformedData.persons,
          stats: transformedData.stats,
          loading: false,
          error: null,
          mode: results.mode,
        });

        setHasCurrentResults(true);
        setActiveTab("current");

        toast.dismiss(searchingToast);

        if (transformedData.persons.length > 0) {
          toast.success(
            `Found ${transformedData.persons.length} matching results`,
          );
        } else {
          toast.info("No matches found for your search criteria");
        }
      }
    } catch (error) {
      console.error("Search failed:", error);
      toast.dismiss(searchingToast);
      toast.error(`Search failed: ${error.message}`);
    }
  };

  const handleViewHistoryResult = (entry) => {
    setViewingHistoryEntry(entry);
    setActiveTab("history-view");
  };

  const handleBackToCurrent = () => {
    setViewingHistoryEntry(null);
    setActiveTab("current");
  };

  const renderContent = () => {
    switch (activeTab) {
      case "upload":
        return <UploadSection onSubmit={handleSubmit} />;

      case "current":
        return (
          <div className="space-y-6">
            {/* ONLY CURRENT RESULTS - NO HISTORY HERE */}
            <div>
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center space-x-2">
                  <div className="w-1 h-6 bg-blue-500 rounded-full"></div>
                  <h2 className="text-xl font-bold text-white">
                    Current Detection Results
                  </h2>
                  {detectionData.persons.length > 0 && (
                    <span className="px-2 py-1 bg-blue-500/20 text-blue-400 rounded-full text-xs">
                      {detectionData.persons.length} items
                    </span>
                  )}
                </div>

                {/* Save to History Button - Only shows when there are results */}
                {detectionData.persons.length > 0 && (
                  <button
                    onClick={() => {
                      addToHistory({
                        videoName: `Manual Save - ${new Date().toLocaleString()}`,
                        results: detectionData.persons,
                        stats: detectionData.stats,
                        mode: detectionData.mode,
                        thumbnail: detectionData.persons[0]?.thumbnail,
                        timestamp: new Date().toISOString(),
                      });
                      toast.success("Results saved to history");
                    }}
                    className="px-3 py-1.5 bg-purple-500/20 text-purple-400 rounded-lg hover:bg-purple-500/30 text-sm flex items-center space-x-1"
                  >
                    <svg
                      className="h-4 w-4"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M8 7H5a2 2 0 00-2 2v9a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-3m-1 4l-3 3m0 0l-3-3m3 3V4"
                      />
                    </svg>
                    <span>Save to History</span>
                  </button>
                )}
              </div>

              {jobPolling && detectionData.persons.length === 0 && (
                <div className="bg-blue-500/10 border border-blue-500/30 rounded-xl p-6 text-center">
                  <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mb-4"></div>
                  <p className="text-blue-400 font-medium">
                    Processing video in background...
                  </p>
                  <p className="text-sm text-gray-400 mt-2">
                    Job ID: {currentJobId}
                  </p>
                  <p className="text-xs text-gray-500 mt-1">
                    This may take a few minutes depending on video length
                  </p>
                </div>
              )}

              {detectionData.loading &&
                !jobPolling &&
                detectionData.persons.length === 0 && (
                  <div className="text-center py-12 bg-gray-800/50 rounded-xl border border-gray-700">
                    <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
                    <p className="text-gray-400 mt-4">
                      Loading detection data...
                    </p>
                  </div>
                )}

              {detectionData.error && (
                <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-6 text-center">
                  <p className="text-red-400">Error: {detectionData.error}</p>
                </div>
              )}

              {!detectionData.loading &&
                !jobPolling &&
                detectionData.persons.length === 0 &&
                !detectionData.error && (
                  <div className="bg-gray-800/50 border border-gray-700 rounded-xl p-12 text-center">
                    <svg
                      className="w-16 h-16 mx-auto text-gray-600 mb-4"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                      />
                    </svg>
                    <h3 className="text-xl font-medium text-gray-300 mb-2">
                      No Current Detections
                    </h3>
                    <p className="text-gray-500">
                      Upload a video to see results here
                    </p>
                  </div>
                )}

              {detectionData.persons.length > 0 && (
                <>
                  <DetectionStats
                    stats={detectionData.stats}
                    mode={detectionData.mode}
                  />
                  <div className="mt-6">
                    <DetectedPersonsTable persons={detectionData.persons} />
                  </div>
                </>
              )}
            </div>
          </div>
        );

      case "history":
        return (
          <div className="space-y-6">
            {/* ONLY HISTORY - NO CURRENT RESULTS HERE */}
            <div>
              <div className="flex items-center mb-4">
                <div className="w-1 h-6 bg-purple-500 rounded-full mr-2"></div>
                <h2 className="text-xl font-bold text-white">Upload History</h2>
                {uploadHistory.length > 0 && (
                  <span className="ml-2 px-2 py-1 bg-purple-500/20 text-purple-400 rounded-full text-xs">
                    {uploadHistory.length} total
                  </span>
                )}
              </div>
              <PreviousResultsTable onViewResult={handleViewHistoryResult} />
            </div>
          </div>
        );

      case "history-view":
        return (
          <div className="space-y-6">
            <button
              onClick={handleBackToCurrent}
              className="mb-4 px-4 py-2 bg-gray-700 text-white rounded-lg hover:bg-gray-600 flex items-center space-x-2"
            >
              <svg
                className="h-4 w-4"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M10 19l-7-7m0 0l7-7m-7 7h18"
                />
              </svg>
              <span>Back to Current Results</span>
            </button>

            {viewingHistoryEntry && (
              <div className="space-y-6">
                <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-lg p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <h2 className="text-xl font-bold text-white mb-1">
                        History View: {viewingHistoryEntry.videoName}
                      </h2>
                      <p className="text-sm text-gray-400">
                        {new Date(
                          viewingHistoryEntry.timestamp,
                        ).toLocaleString()}
                      </p>
                    </div>
                    {viewingHistoryEntry.prompt && (
                      <span className="px-3 py-1 bg-purple-500/20 text-purple-400 rounded-full text-sm">
                        Prompt: {viewingHistoryEntry.prompt}
                      </span>
                    )}
                  </div>
                </div>

                <DetectionStats
                  stats={viewingHistoryEntry.stats}
                  mode={viewingHistoryEntry.mode}
                />

                <DetectedPersonsTable
                  persons={viewingHistoryEntry.results || []}
                />
              </div>
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
        <Navigation
          activeTab={activeTab}
          onTabChange={handleTabChange}
          isUploading={isUploading}
          uploadProgress={uploadProgress}
        />
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
      toast.success("Successfully logged in!");
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
      toast.success("Registration successful! Please log in.");
      navigate("/login");
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
        <UploadProvider>
          <HistoryProvider>
            <Toaster
              position="top-right"
              toastOptions={{
                duration: 4000,
                style: {
                  background: "#1f2937",
                  color: "#f3f4f6",
                  border: "1px solid #374151",
                },
                success: {
                  duration: 3000,
                  iconTheme: {
                    primary: "#10b981",
                    secondary: "#f3f4f6",
                  },
                },
                error: {
                  duration: 4000,
                  iconTheme: {
                    primary: "#ef4444",
                    secondary: "#f3f4f6",
                  },
                },
                loading: {
                  duration: Infinity,
                  style: {
                    background: "#1f2937",
                    color: "#f3f4f6",
                  },
                },
              }}
            />
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
          </HistoryProvider>
        </UploadProvider>
      </AuthProvider>
    </Router>
  );
}

export default App;
