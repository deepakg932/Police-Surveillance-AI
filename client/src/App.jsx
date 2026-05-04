import React, { useState, useEffect } from "react";
import WorkflowProgress from "./components/Detection/WorkflowProgress";
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
    liveResults: [],
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
  const [hasCurrentResults, setHasCurrentResults] = useState(false);
  const { logout, user, authFetch } = useAuth();
  const { isUploading, uploadProgress, resetUpload } = useUpload();
  const { uploadHistory, refreshHistory } = useHistory();
  const handleTabChange = async (tabId) => {
    setActiveTab(tabId);

    if (tabId !== "history-view") {
      setViewingHistoryEntry(null);
    }

    if (tabId === "history") {
      await refreshHistory();
    }
  };
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
          confidence: item.confidence,
          thumbnail: imageUrl,
          image_path: item.image_path,
          screenshotUrl: item.screenshotUrl,
          bbox: item.bbox,
          timestamp: item.timestamp,
          fullImageUrl: imageUrl,
          processing_time: item.processing_time,
          ocrText: item.ocrText,
          ...item,
        };
      });

      const stats = {
        totalPersons: apiData.totalPersons || 0,
        totalHelmet: apiData.totalHelmet || 0,
        totalVehicles: apiData.totalVehicles || 0,
        totalDetections: apiData.totalDetections || persons.length,
        totalCommonPersons:
          typeof apiData.totalCommonPersons === "number"
            ? apiData.totalCommonPersons
            : new Set(persons.map((p) => p.commonPersonId).filter(Boolean))
                .size,
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
        totalCommonPersons: apiData.totalCommonPersons || 0,
        processingTime: apiData.processing_time || "N/A",
        filters: apiData.filters || {},
      },
    };
  };

  // Function to refresh current detection data
  const refreshCurrentDetectionData = async () => {
    if (currentJobId) {
      // If we have a job ID, refetch its status
      try {
        const response = await authFetch(
          `${API_BASE_URL}/video/status/${currentJobId}`,
        );
        if (response.ok) {
          const statusData = await response.json();
          if (statusData.status === "completed" && statusData.results) {
            const transformedData = transformApiData({
              results: statusData.results,
              processing_time: statusData.processing_time || "N/A",
            });
            setDetectionData({
              persons: transformedData.persons,
              stats: transformedData.stats,
              loading: false,
              error: null,
              mode: detectionData.mode,
            });
          }
        }
      } catch (error) {
        console.error("Error refreshing detection data:", error);
      }
    }
  };

  // Delete detection handlers
  // Delete detection handlers - FIXED VERSION
  const handleDeleteSingleDetection = async (trackingId) => {
    if (!trackingId) return false;
    try {
      // ✅ currentJobId bhi bhejo
      const url = currentJobId
        ? `${API_BASE_URL}/video/delete/${trackingId}?jobId=${currentJobId}`
        : `${API_BASE_URL}/video/delete/${trackingId}`;

      const response = await authFetch(url, { method: "DELETE" });
      if (!response.ok) throw new Error(`Delete failed: ${response.status}`);

      setDetectionData((prev) => {
        const updatedPersons = prev.persons.filter(
          (p) => p.trackingId !== trackingId,
        );
        if (updatedPersons.length === 0) {
          return {
            persons: [],
            liveResults: [],
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
          };
        }

        return {
          ...prev,
          persons: updatedPersons,
          stats: { ...prev.stats, totalDetections: updatedPersons.length },
        };
      });

      toast.success("Detection deleted");
      return true;
    } catch (error) {
      toast.error(`Failed to delete: ${error.message}`);
      return false;
    }
  };

  // Add a new handler for delete all that clears everything
  const handleDeleteAllDetections = async () => {
    try {
      // ✅ currentJobId bhejo — sirf current job/batch ki detections delete hongi
      const url = currentJobId
        ? `${API_BASE_URL}/video/deleteall?jobId=${currentJobId}`
        : `${API_BASE_URL}/video/deleteall`;

      const response = await authFetch(url, { method: "DELETE" });
      if (!response.ok) throw new Error(`Delete failed: ${response.status}`);

      setDetectionData({
        persons: [],
        liveResults: [],
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

      toast.success("All detections deleted");
      return true;
    } catch (error) {
      toast.error(`Failed: ${error.message}`);
      return false;
    }
  };

  const handleDeleteDetections = async (trackingIds) => {
    if (!trackingIds || trackingIds.length === 0) return false;

    let successCount = 0;
    const successfulDeletes = [];

    for (const trackingId of trackingIds) {
      try {
        // ✅ currentJobId bhi bhejo
        const url = currentJobId
          ? `${API_BASE_URL}/video/delete/${trackingId}?jobId=${currentJobId}`
          : `${API_BASE_URL}/video/delete/${trackingId}`;

        const response = await authFetch(url, { method: "DELETE" });
        if (response.ok) {
          successCount++;
          successfulDeletes.push(trackingId);
        }
      } catch (error) {
        console.error(`Error deleting ${trackingId}:`, error);
      }
    }

    if (successCount > 0) {
      setDetectionData((prev) => {
        const updatedPersons = prev.persons.filter(
          (p) => !successfulDeletes.includes(p.trackingId),
        );

        if (updatedPersons.length === 0) {
          return {
            persons: [],
            liveResults: [],
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
          };
        }
        return {
          ...prev,
          persons: updatedPersons,
          stats: { ...prev.stats, totalDetections: updatedPersons.length },
        };
      });
      toast.success(`Deleted ${successCount} detections`);
      return true;
    }
    return false;
  };

  useEffect(() => {
    return () => {
      if (detectionData.videoPreviewUrl) {
        URL.revokeObjectURL(detectionData.videoPreviewUrl);
      }
    };
  }, [detectionData.videoPreviewUrl]);

  const pollJobStatus = async (jobId, videoFile, formInput) => {
    setJobPolling(true);
    setCurrentJobId(jobId);
    setHasCurrentResults(false);
    setDetectionData({
      prompt: formInput.text || "",
      queryImagePreviewUrl: formInput.image
        ? URL.createObjectURL(formInput.image)
        : "",
      fileName: videoFile?.name || "CCTV_Footage.mp4",
      videoPreviewUrl: videoFile ? URL.createObjectURL(videoFile) : "",
      persons: [],
      liveResults: [],
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

    const maxAttempts = 1440; // 2 hours with 5s polling interval
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

          const transformedData = transformApiData({
            results: statusData.results || [],
            processing_time: statusData.processing_time || "N/A",
            mode: formInput.text ? "Text Search" : "Video Upload",
            totalDetections: statusData.totalResults || 0,
          });

          await refreshHistory();

          setDetectionData({
            prompt: formInput.text || "",
            queryImagePreviewUrl: formInput.image
              ? URL.createObjectURL(formInput.image)
              : "",
            fileName: videoFile?.name || "CCTV_Footage.mp4",
            videoPreviewUrl: videoFile ? URL.createObjectURL(videoFile) : "",
            persons: transformedData.persons,
            stats: transformedData.stats,
            liveResults: [],
            loading: false,
            error: null,
            mode: formInput.text ? "Text Search" : "Video Upload",
          });

          setHasCurrentResults(true);
          setJobPolling(false);
          setCurrentJobId(null);
          setActiveTab("current");

          toast.success(
            `✅ Processing complete! Found ${transformedData.persons.length} detection${transformedData.persons.length !== 1 ? "s" : ""}.`,
          );
          return;
        }

        if (statusData.status === "failed") {
          clearInterval(interval);

          setDetectionData({
            persons: [],
            liveResults: [],
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
          setActiveTab("upload");
          setCurrentJobId(null);
          setHasCurrentResults(false);

          toast.error(
            `Processing failed: ${statusData.errorMessage || "Unknown error"}`,
          );
          return;
        }

        if (statusData.status === "processing") {
          const transformedData = transformApiData({
            results: statusData.results || [],
            processing_time: statusData.processing_time || "N/A",
            totalDetections: statusData.total || 0,
          });

          setDetectionData((prev) => ({
            ...prev,
            liveResults: transformedData.persons,
            loading: true,
            error: null,
          }));
        }

        if (attempts >= maxAttempts) {
          clearInterval(interval);

          setDetectionData({
            persons: [],
            liveResults: [],
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
          setActiveTab("upload");
          toast.error(
            "Processing is taking too long. Please check again later.",
          );
        }
      } catch (error) {
        console.error("Polling error:", error);
        clearInterval(interval);

        try {
          await authFetch(`${API_BASE_URL}/video/status/${jobId}/fail`, {
            method: "POST",
          });

          await refreshHistory();
        } catch (failErr) {
          console.error("Failed to mark failed:", failErr);
        }

        setDetectionData({
          persons: [],
          liveResults: [],
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
        setActiveTab("history"); // ✅ direct history me failed status dikhega

        toast.error(`Processing failed: ${error.message}`);
      }
    }, 5000);
  };

  const pollBatchStatus = async (batchId, formInput, selectedVideos = []) => {
    setJobPolling(true);
    setCurrentJobId(batchId);
    setHasCurrentResults(false);
    setDetectionData({
      persons: [],
      liveResults: [],
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
      prompt: formInput.text || "",
      queryImagePreviewUrl: formInput.image
        ? URL.createObjectURL(formInput.image)
        : "",
      fileName: selectedVideos?.[0]?.name || "Multiple Videos",
      videoPreviewUrl:
        selectedVideos && selectedVideos.length > 0
          ? URL.createObjectURL(selectedVideos[0])
          : "",
      videoList: selectedVideos || [],
      currentVideoIndex: 0,
      currentVideoFile: selectedVideos?.[0] || null,
    });

    const maxAttempts = 1440;
    let attempts = 0;

    const interval = setInterval(async () => {
      attempts += 1;

      try {
        // ✅ Batch status check — single video jaisa hi
        const response = await authFetch(
          `${API_BASE_URL}/video/batch/${batchId}`,
        );

        if (!response.ok) {
          throw new Error(`Batch status check failed: ${response.status}`);
        }

        const statusData = await response.json();
        console.log("Batch status:", statusData);

        // 🔥 LIVE RESULTS DURING PROCESSING (MULTI VIDEO)
        if (statusData.status === "processing") {
          const transformedData = transformApiData({
            results: statusData.results || [],
            processing_time: statusData.processing_time || "N/A",
            totalDetections: statusData.totalResults || 0,
          });

          setDetectionData((prev) => ({
            ...prev,
            liveResults: transformedData.persons,
            loading: true,
            error: null,
          }));
        }
        // Progress dikhao — kitni videos complete hui
        if (statusData.totalVideos > 0) {
          const completedCount = Number(statusData.completed || 0);
          const totalVideos = Number(
            statusData.totalVideos || selectedVideos.length,
          );

          const runningIndex = statusData.jobs?.findIndex(
            (j) => j.status === "processing",
          );

          const currentVideoIndex =
            runningIndex >= 0
              ? runningIndex
              : Math.min(completedCount, selectedVideos.length - 1);

          const currentVideoFile = selectedVideos[currentVideoIndex];

          const progressMsg = `Processing video ${currentVideoIndex + 1}/${totalVideos} — completed ${completedCount}/${totalVideos}`;

          setDetectionData((prev) => ({
            ...prev,
            loading: true,
            error: null,
            progressMsg,
            currentVideoIndex,
            currentVideoFile,
            fileName:
              currentVideoFile?.name || `Video ${currentVideoIndex + 1}`,
            videoPreviewUrl: currentVideoFile
              ? URL.createObjectURL(currentVideoFile)
              : "",
            videoList: selectedVideos,
          }));
        }

        if (
          statusData.status === "failed" ||
          (statusData.failed > 0 && statusData.completed === 0)
        ) {
          clearInterval(interval);

          await refreshHistory();

          setDetectionData({
            persons: [],
            liveResults: [],
            stats: {
              totalPersons: 0,
              totalHelmet: 0,
              totalVehicles: 0,
              totalDetections: 0,
              processingTime: "N/A",
              filters: {},
            },
            loading: false,
            error:
              statusData.errorMessage ||
              "Python server unreachable / network failed",
            mode: null,
          });

          setJobPolling(false);
          setCurrentJobId(null);
          setHasCurrentResults(false);

          setActiveTab("upload");

          toast.error(
            `Processing failed: ${
              statusData.errorMessage ||
              "Python server unreachable / network failed"
            }`,
          );

          return;
        }

        // ✅ Sab videos complete
        if (statusData.isAllDone || statusData.status === "completed") {
          clearInterval(interval);

          // Single video jaisa hi transform karo
          const transformedData = transformApiData({
            results: statusData.results || [],
            processing_time: statusData.processing_time || "N/A",
            mode: formInput.text ? "Text Search" : "Video Upload",
            totalDetections: statusData.totalResults || 0,
          });

          await refreshHistory();

          setDetectionData({
            prompt: formInput.text || "",
            queryImagePreviewUrl: formInput.image
              ? URL.createObjectURL(formInput.image)
              : "",
            fileName:
              selectedVideos?.[selectedVideos.length - 1]?.name ||
              "Multiple Videos",
            videoPreviewUrl:
              selectedVideos?.length > 0
                ? URL.createObjectURL(selectedVideos[selectedVideos.length - 1])
                : "",
            videoList: selectedVideos || [],
            currentVideoIndex: selectedVideos.length - 1,
            currentVideoFile:
              selectedVideos?.[selectedVideos.length - 1] || null,
            progressMsg: `Processing... ${selectedVideos.length}/${selectedVideos.length} videos done`,
            persons: transformedData.persons,
            stats: transformedData.stats,
            loading: false,
            liveResults: [],
            error: null,
            mode: formInput.text ? "Text Search" : "Video Upload",
          });

          setHasCurrentResults(true);
          setJobPolling(false);
          setCurrentJobId(null);
          setActiveTab("current");

          toast.success(
            `✅ Sab videos process ho gaye! ${transformedData.persons.length} detection${transformedData.persons.length !== 1 ? "s" : ""} mili.`,
          );
          return;
        }

        if (attempts >= maxAttempts) {
          clearInterval(interval);
          setJobPolling(false);
          setCurrentJobId(null);
          toast.error("Processing timeout ho gaya");
        }
      } catch (error) {
        console.error("Batch polling error:", error);
        clearInterval(interval);
        setDetectionData({
          persons: [],
          liveResults: [],
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
        toast.error(`Polling failed: ${error.message}`);
      }
    }, 5000);
  };

  const handleSubmit = async (data) => {
    console.log("Submitting:", data);
    let uploadingToast = null;

    try {
      const selectedVideos = Array.isArray(data.videos)
        ? data.videos
        : [data.file || data.video].filter(Boolean);

      if (!selectedVideos.length) {
        toast.error("No video file selected");
        return;
      }

      const formData = new FormData();
      const isMulti = selectedVideos.length > 1;

      if (isMulti) {
        // ✅ Multi video — "files" key se bhejo
        selectedVideos.forEach((videoFile) => {
          formData.append("files", videoFile, videoFile.name);
        });
      } else {
        // ✅ Single video — "file" key se bhejo (same as before)
        formData.append("file", selectedVideos[0], selectedVideos[0].name);
      }

      if (data.image) {
        formData.append("image", data.image, data.image.name);
      }
      if (data.text) {
        formData.append("text", data.text);
      }

      uploadingToast = toast.loading(
        isMulti
          ? `Uploading ${selectedVideos.length} videos...`
          : "Uploading video...",
      );

      // ✅ Single ya Multi — alag endpoint
      const endpoint = isMulti
        ? `${API_BASE_URL}/video/upload/multi`
        : `${API_BASE_URL}/video/upload`;

      const response = await authFetch(endpoint, {
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
      resetUpload();

      // ✅ Single video → jobId se poll karo (same as before)
      if (result.jobId && !result.batchId) {
        toast.success("Upload successful. Processing started.");
        setActiveTab("current");
        pollJobStatus(result.jobId, selectedVideos[0], data);
        return;
      }

      // ✅ Multi video → batchId se poll karo
      if (result.batchId) {
        toast.success(
          `${selectedVideos.length} videos upload ho gaye. Processing shuru...`,
        );
        setActiveTab("current");
        pollBatchStatus(result.batchId, data, selectedVideos); // ← naya function
        return;
      }

      // Fallback (agar seedha results aaye)
      const transformedData = transformApiData(result);
      await refreshHistory();
      setDetectionData({
        persons: transformedData.persons,
        stats: transformedData.stats,
        loading: false,
        error: null,
        mode: result.mode,
      });
      setHasCurrentResults(true);
      setActiveTab("current");
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

  const handleDeleteHistoryViewSingleDetection = async (trackingId) => {
    if (!trackingId || !viewingHistoryEntry?.id) return false;
    try {
      const response = await authFetch(
        `${API_BASE_URL}/video/delete/${trackingId}?jobId=${encodeURIComponent(viewingHistoryEntry.id)}`,
        { method: "DELETE" },
      );
      if (!response.ok) {
        throw new Error(`Delete failed: ${response.status}`);
      }

      setViewingHistoryEntry((prev) => {
        if (!prev) return prev;
        const updatedResults = (prev.results || []).filter(
          (item) => item.trackingId !== trackingId,
        );
        return {
          ...prev,
          results: updatedResults,
          stats: {
            ...(prev.stats || {}),
            totalDetections: updatedResults.length,
          },
        };
      });
      await refreshHistory();
      toast.success(`Deleted detection ${trackingId}`);
      return true;
    } catch (error) {
      toast.error(`Failed to delete: ${error.message}`);
      return false;
    }
  };

  const handleDeleteHistoryViewDetections = async (trackingIds = []) => {
    if (!trackingIds.length || !viewingHistoryEntry?.id) return false;
    let successCount = 0;
    for (const trackingId of trackingIds) {
      const ok = await handleDeleteHistoryViewSingleDetection(trackingId);
      if (ok) successCount += 1;
    }
    return successCount > 0;
  };

  const handleBackToHistory = () => {
    setViewingHistoryEntry(null);
    setActiveTab("history");
  };

  const renderContent = () => {
    switch (activeTab) {
      case "upload":
        return <UploadSection onSubmit={handleSubmit} />;

      case "current":
        return (
          <div className="space-y-6">
            <WorkflowProgress
              job={{
                jobId: currentJobId,
                fileName: detectionData.fileName || "CCTV_Footage.mp4",
                textNote: detectionData.prompt || "",
                queryImagePreviewUrl: detectionData.queryImagePreviewUrl || "",
                videoPreviewUrl: detectionData.videoPreviewUrl || "",
                videoList: detectionData.videoList || [],
                currentVideoIndex: detectionData.currentVideoIndex || 0,
                currentVideoFile: detectionData.currentVideoFile || null,
                progressMsg: detectionData.progressMsg || "",
                status: jobPolling
                  ? "processing"
                  : detectionData.error
                    ? "failed"
                    : detectionData.persons.length > 0
                      ? "completed"
                      : "idle",
              }}
              results={
                jobPolling
                  ? detectionData.liveResults || []
                  : detectionData.persons || []
              }
              status={
                jobPolling
                  ? "processing"
                  : detectionData.error
                    ? "failed"
                    : detectionData.persons.length > 0
                      ? "completed"
                      : "idle"
              }
            />

            {!jobPolling && detectionData.persons.length > 0 && (
              <>
                <DetectionStats
                  stats={detectionData.stats}
                  mode={detectionData.mode}
                />

                <DetectedPersonsTable
                  persons={detectionData.persons}
                  onDeleteSingleDetection={handleDeleteSingleDetection}
                  onDeleteDetections={handleDeleteDetections}
                  onDeleteAllDetections={handleDeleteAllDetections}
                  onDeleteComplete={refreshCurrentDetectionData}
                  apiBaseUrl={API_BASE_URL}
                  authFetch={authFetch}
                />
              </>
            )}
          </div>
        );

      case "history":
        return (
          <div className="space-y-6">
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
              onClick={handleBackToHistory}
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
              <span>Back to History</span>
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
                  persons={(viewingHistoryEntry.results || []).map((r) => ({
                    ...r,
                    processingTime:
                      r.processingTime ||
                      r.processing_time ||
                      viewingHistoryEntry.processing_time ||
                      "N/A",
                    processing_time:
                      r.processing_time ||
                      r.processingTime ||
                      viewingHistoryEntry.processing_time ||
                      "N/A",
                  }))}
                  // persons={viewingHistoryEntry.results || []}
                  onDeleteSingleDetection={
                    handleDeleteHistoryViewSingleDetection
                  }
                  onDeleteDetections={handleDeleteHistoryViewDetections}
                  apiBaseUrl={API_BASE_URL}
                  authFetch={authFetch}
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
