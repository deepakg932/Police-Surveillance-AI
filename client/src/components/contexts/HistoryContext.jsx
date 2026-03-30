import React, { createContext, useState, useContext, useEffect } from 'react';
import { useAuth } from './AuthContext';

const HistoryContext = createContext(null);

export const useHistory = () => {
  const context = useContext(HistoryContext);
  if (!context) {
    throw new Error('useHistory must be used within a HistoryProvider');
  }
  return context;
};

export const HistoryProvider = ({ children }) => {
  const { user } = useAuth();
  const [uploadHistory, setUploadHistory] = useState([]);
  const [currentResults, setCurrentResults] = useState(null);
  const [loading, setLoading] = useState(false);

  // Load history from localStorage on mount and when user changes
  useEffect(() => {
    if (user) {
      loadHistory();
    } else {
      setUploadHistory([]);
    }
  }, [user]);

  const loadHistory = () => {
    try {
      const savedHistory = localStorage.getItem(`upload_history_${user?.id}`);
      if (savedHistory) {
        setUploadHistory(JSON.parse(savedHistory));
      }
    } catch (err) {
      console.error('Error loading history:', err);
    }
  };

  // Add new upload to history
  const addToHistory = (uploadData) => {
    if (!user) return null;

    const newEntry = {
      id: `upload_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      timestamp: new Date().toISOString(),
      videoName: uploadData.videoName || 'Untitled',
      videoFile: uploadData.videoFile ? {
        name: uploadData.videoFile.name,
        size: uploadData.videoFile.size,
        type: uploadData.videoFile.type,
      } : null,
      prompt: uploadData.prompt || null,
      image: uploadData.image ? {
        name: uploadData.image.name,
        size: uploadData.image.size,
        type: uploadData.image.type
      } : null,
      text: uploadData.text || null,
      results: uploadData.results || [],
      stats: uploadData.stats || {},
      mode: uploadData.mode || null,
      processingTime: uploadData.processingTime || 'N/A',
      thumbnail: uploadData.thumbnail || null
    };

    const updatedHistory = [newEntry, ...uploadHistory].slice(0, 50); // Keep last 50
    setUploadHistory(updatedHistory);
    
    // Save to localStorage
    localStorage.setItem(`upload_history_${user.id}`, JSON.stringify(updatedHistory));
    
    return newEntry;
  };

  // Set current results
  const setCurrent = (results) => {
    setCurrentResults(results);
  };

  // Clear current results
  const clearCurrent = () => {
    setCurrentResults(null);
  };

  // Clear all history
  const clearHistory = () => {
    if (!user) return;
    setUploadHistory([]);
    localStorage.removeItem(`upload_history_${user.id}`);
  };

  // Delete specific entry
  const deleteHistoryEntry = (entryId) => {
    if (!user) return;
    
    const updatedHistory = uploadHistory.filter(entry => entry.id !== entryId);
    setUploadHistory(updatedHistory);
    localStorage.setItem(`upload_history_${user.id}`, JSON.stringify(updatedHistory));
  };

  // Get entry by ID
  const getHistoryEntry = (entryId) => {
    return uploadHistory.find(entry => entry.id === entryId);
  };

  return (
    <HistoryContext.Provider value={{
      uploadHistory,
      currentResults,
      loading,
      addToHistory,
      setCurrent,
      clearCurrent,
      clearHistory,
      deleteHistoryEntry,
      getHistoryEntry
    }}>
      {children}
    </HistoryContext.Provider>
  );
};