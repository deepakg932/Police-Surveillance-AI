// components/contexts/HistoryContext.jsx
import React, {
  createContext,
  useState,
  useContext,
  useEffect,
  useCallback,
} from "react";
import { useAuth } from "./AuthContext";

const HistoryContext = createContext(null);

export const useHistory = () => {
  const context = useContext(HistoryContext);
  if (!context) {
    throw new Error("useHistory must be used within a HistoryProvider");
  }
  return context;
};

export const HistoryProvider = ({ children }) => {
  const { user, authFetch } = useAuth();
  const [uploadHistory, setUploadHistory] = useState([]);
  const [currentResults, setCurrentResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const API_BASE_URL = import.meta.env.VITE_API_BASE_URL;

  // Fetch history from backend API
  const fetchHistory = useCallback(async () => {
    if (!user) return [];

    setLoading(true);
    setError(null);

    try {
      const response = await authFetch(`${API_BASE_URL}/video/history`);

      if (!response.ok) {
        throw new Error(`Failed to fetch history: ${response.status}`);
      }

      const data = await response.json();
      console.log("History data from API:", data);

      // Transform API response to match our frontend format
      const transformedHistory = transformHistoryData(data);
      setUploadHistory(transformedHistory);

      return transformedHistory;
    } catch (err) {
      console.error("Error fetching history:", err);
      setError(err.message);
      // API-only mode: do not fallback to any local/static cache.
      setUploadHistory([]);
      return [];
    } finally {
      setLoading(false);
    }
  }, [user, authFetch, API_BASE_URL]);

  // Transform API history data to frontend format
  const transformHistoryData = (apiData) => {
    if (!apiData?.history || !Array.isArray(apiData.history)) return [];

    return apiData.history.map((item, index) => {
      const rawResults = Array.isArray(item.results) ? item.results : [];

      const transformedDetections = rawResults.map((detection, idx) => ({
        id: detection.trackingId || `detection_${idx}`,
        trackingId: detection.trackingId || `track_${idx}`,
        object: detection.object || "unknown",
        confidence: detection.confidence || 0,
        confidencePercentage: detection.confidence
          ? Math.round(detection.confidence * 100)
          : 0,
        timestamp: detection.timestamp || 0,
        image_path: detection.image_path || detection.imagePath || "",
        imagepath_full: detection.image_path || detection.imagePath || "",
        screenshotUrl: detection.screenshotUrl || "",
        thumbnail: detection.screenshotUrl || detection.image_path || "",
        bbox: detection.bbox || [],
        color: detection.color || "",
        ocrText: detection.ocrText || "",
      }));

      const processingTime = item.processing_time || item.processingTime || "0";

      const total =
        item.total_found ?? item.total ?? transformedDetections.length ?? 0;

      return {
        id: item.jobId || item.id || `history_${index}`,
        jobId: item.jobId || item.id || `history_${index}`,
        timestamp: item.timestamp || new Date().toISOString(),
        videoName:
          item.videoName || item.videoUrl?.split("/").pop() || "Untitled Video",
        videoUrl: item.videoUrl || "",
        prompt: item.prompt || "",
        text: item.prompt || "",

        // ✅ Direct API results show karo
        results: transformedDetections,

        processingTime,
        processing_time: processingTime,
        status: item.status || "completed",
        total,
        total_found: total,
        totalUniqueObjects: item.totalUniqueObjects || total,
        counts: item.counts || {},

        mode: item.prompt ? "Text Search" : "Video Upload",

        stats: {
          totalDetections: total,
          totalPersons: 0,
          totalVehicles: total,
          totalHelmet: 0,
          processingTime,
          filters: { prompt: item.prompt || "none" },
        },
      };
    });
  };

  // Calculate statistics from detections
  const calculateStats = (detections, item) => {
    const totalDetections = detections.length || item.totalDetections || 0;

    // Count unique objects
    const objectCounts = {};
    detections.forEach((det) => {
      const objType = det.object || "unknown";
      objectCounts[objType] = (objectCounts[objType] || 0) + 1;
    });

    return {
      totalDetections: totalDetections,
      totalPersons: objectCounts["person"] || 0,
      totalVehicles:
        (objectCounts["car"] || 0) +
        (objectCounts["scooter"] || 0) +
        (objectCounts["bike"] || 0) +
        (objectCounts["motorcycle"] || 0),
      totalHelmet: objectCounts["helmet"] || 0,
      processingTime: item.processingTime || "0",
      filters: { prompt: item.prompt || "none" },
    };
  };

  // Load history on mount and when user changes
  useEffect(() => {
    if (user) {
      fetchHistory();
    } else {
      setUploadHistory([]);
    }
  }, [user, fetchHistory]);

  // Delete single history entry
  const deleteHistoryEntry = async (id) => {
    try {
      const response = await authFetch(
        `${API_BASE_URL}/video/history/${id}`, // ← sahi hai ye
        { method: "DELETE" },
      );
      if (!response.ok) throw new Error("Failed");
      await refreshHistory();
      return true;
    } catch (err) {
      return false;
    }
  };

  // Delete multiple history entries
  const deleteMultipleHistoryEntries = async (ids) => {
    try {
      const response = await authFetch(
        `${API_BASE_URL}/video/history/batch`, // ← /batch lagao
        {
          method: "DELETE",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ ids }),
        },
      );
      if (!response.ok) throw new Error("Failed");
      await refreshHistory();
      return { successCount: ids.length, failCount: 0 };
    } catch (err) {
      return { successCount: 0, failCount: ids.length };
    }
  };
  // Clear all history
  const clearHistory = async () => {
    try {
      const response = await authFetch(
        `${API_BASE_URL}/video/history/clear`, // ← /clear lagao
        { method: "DELETE" },
      );
      if (!response.ok) throw new Error("Failed");
      await refreshHistory();
      return true;
    } catch (err) {
      return false;
    }
  };

  // Set current results
  const setCurrent = (results) => {
    setCurrentResults(results);
  };

  // Clear current results
  const clearCurrent = () => {
    setCurrentResults(null);
  };

  // Get entry by ID
  const getHistoryEntry = (entryId) => {
    return uploadHistory.find((entry) => entry.id === entryId);
  };

  // Refresh history manually
  const refreshHistory = async () => {
    return await fetchHistory();
  };

  return (
    <HistoryContext.Provider
      value={{
        uploadHistory,
        currentResults,
        loading,
        error,
        setCurrent,
        clearCurrent,
        clearHistory,
        deleteHistoryEntry,
        deleteMultipleHistoryEntries,
        getHistoryEntry,
        refreshHistory,
      }}
    >
      {children}
    </HistoryContext.Provider>
  );
};
