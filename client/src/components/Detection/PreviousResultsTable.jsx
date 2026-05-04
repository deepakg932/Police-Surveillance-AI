// components/Detection/PreviousResultsTable.jsx
import React, { useState, useEffect } from "react";
import { toast } from "react-hot-toast";
import {
  Clock,
  Video,
  FileText,
  ChevronDown,
  ChevronUp,
  Trash2,
  Eye,
  Calendar,
  Filter,
  RefreshCw,
  Loader2,
  AlertCircle,
  CheckCircle,
  XCircle,
  Search,
  X,
} from "lucide-react";
import { useHistory } from "../contexts/HistoryContext";
import InlineConfirmPopover from "../Common/InlineConfirmPopover";

const PreviousResultsTable = ({ onViewResult }) => {
  const {
    uploadHistory,
    deleteHistoryEntry,
    deleteMultipleHistoryEntries,
    clearHistory,
    loading,
    error,
    refreshHistory,
  } = useHistory();
  const [expandedId, setExpandedId] = useState(null);
  const [filterText, setFilterText] = useState("");
  const [sortOrder, setSortOrder] = useState("desc");
  const [selectedEntries, setSelectedEntries] = useState([]);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [searchField, setSearchField] = useState("all"); // 'all', 'videoName', 'prompt', 'status'
  const [dateFilter, setDateFilter] = useState("all"); // 'all', 'today', 'week', 'month'
  const [viewMode, setViewMode] = useState("detailed"); // 'detailed' | 'compact'
  const [confirmState, setConfirmState] = useState({
    key: null,
    message: "",
    onConfirm: null,
  });
  const API_BASE_URL = import.meta.env.VITE_API_BASE_URL;

  useEffect(() => {
    const hasProcessing = uploadHistory.some((h) => h.status === "processing");
    if (!hasProcessing) return;

    const timer = setInterval(() => {
      refreshHistory();
    }, 3000);

    return () => clearInterval(timer);
  }, [uploadHistory, refreshHistory]);

  const resolveVideoUrl = (entry) => {
    const rawUrl = entry?.videoUrl;
    if (!rawUrl || typeof rawUrl !== "string") return "";
    if (/^https?:\/\//i.test(rawUrl)) return rawUrl;

    const baseUrl = (API_BASE_URL || "").replace(/\/+$/, "");
    if (!baseUrl) return "";
    return `${baseUrl}/${rawUrl.replace(/^\/+/, "")}`;
  };

  const resolveDetectionPreviewImage = (result) => {
    const direct =
      result?.imagepath_full || result?.image_path || result?.imagePath || "";
    if (typeof direct === "string" && /^https?:\/\//i.test(direct)) {
      return direct;
    }

    const path =
      result?.imagepath_full || result?.image_path || result?.imagePath || "";
    if (!path || typeof path !== "string") return "";
    if (/^https?:\/\//i.test(path)) return path;

    return `https://workingcart.com/files/${path.replace(/^\/+/, "")}`;
  };

  const openConfirm = (key, message, onConfirm) => {
    setConfirmState({ key, message, onConfirm });
  };

  const closeConfirm = () => {
    setConfirmState({ key: null, message: "", onConfirm: null });
  };

  const executeConfirm = async () => {
    const action = confirmState.onConfirm;
    closeConfirm();
    if (typeof action === "function") {
      await action();
    }
  };

  // Format date
  const formatDate = (timestamp) => {
    if (!timestamp) return "Unknown date";

    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 1) return "Just now";
    if (diffMins < 60)
      return `${diffMins} minute${diffMins > 1 ? "s" : ""} ago`;
    if (diffHours < 24)
      return `${diffHours} hour${diffHours > 1 ? "s" : ""} ago`;
    if (diffDays < 7) return `${diffDays} day${diffDays > 1 ? "s" : ""} ago`;

    return date.toLocaleDateString("en-US", {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  // Get status badge
  const getStatusBadge = (status) => {
    switch (status) {
      case "completed":
        return (
          <span className="px-2 py-0.5 bg-green-500/20 text-green-400 rounded-full text-xs flex items-center gap-1">
            <CheckCircle className="h-3 w-3" /> Completed
          </span>
        );
      case "processing":
        return (
          <span className="px-2 py-0.5 bg-yellow-500/20 text-yellow-400 rounded-full text-xs flex items-center gap-1">
            <Loader2 className="h-3 w-3 animate-spin" /> Processing
          </span>
        );
      case "failed":
        return (
          <span className="px-2 py-0.5 bg-red-500/20 text-red-400 rounded-full text-xs flex items-center gap-1">
            <XCircle className="h-3 w-3" /> Failed
          </span>
        );
      default:
        return null;
    }
  };

  // Handle manual refresh
  const handleRefresh = async () => {
    setIsRefreshing(true);
    await refreshHistory();
    setIsRefreshing(false);
  };

  // Filter by date range
  const filterByDate = (timestamp) => {
    if (dateFilter === "all") return true;

    const date = new Date(timestamp);
    const now = new Date();
    const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
    const weekAgo = new Date(today.getTime() - 7 * 24 * 60 * 60 * 1000);
    const monthAgo = new Date(today.getTime() - 30 * 24 * 60 * 60 * 1000);

    switch (dateFilter) {
      case "today":
        return date >= today;
      case "week":
        return date >= weekAgo;
      case "month":
        return date >= monthAgo;
      default:
        return true;
    }
  };

  // Filter and sort history
  const filteredHistory = uploadHistory
    .filter((entry) => {
      // Date filter
      if (!filterByDate(entry.timestamp)) return false;

      // Text search filter
      if (!filterText) return true;

      const searchLower = filterText.toLowerCase();

      switch (searchField) {
        case "videoName":
          return entry.videoName?.toLowerCase().includes(searchLower);
        case "prompt":
          return entry.prompt?.toLowerCase().includes(searchLower);
        case "status":
          return entry.status?.toLowerCase().includes(searchLower);
        default:
          return (
            entry.videoName?.toLowerCase().includes(searchLower) ||
            entry.prompt?.toLowerCase().includes(searchLower) ||
            entry.text?.toLowerCase().includes(searchLower) ||
            entry.mode?.toLowerCase().includes(searchLower) ||
            entry.status?.toLowerCase().includes(searchLower)
          );
      }
    })
    .sort((a, b) => {
      const dateA = new Date(a.timestamp);
      const dateB = new Date(b.timestamp);
      return sortOrder === "desc" ? dateB - dateA : dateA - dateB;
    });

  const toggleExpand = (id) => {
    setExpandedId(expandedId === id ? null : id);
  };

  // Delete single entry with confirmation
  const handleDelete = async (e, id, entryName) => {
    e.stopPropagation();
    openConfirm(
      `history-single-${id}`,
      `Delete "${entryName || "this item"}" from history?`,
      async () => {
        const success = await deleteHistoryEntry(id);
        if (success) {
          // Remove from selected entries if it was selected
          setSelectedEntries((prev) =>
            prev.filter((entryId) => entryId !== id),
          );
          toast.success("Entry deleted successfully");
        }
      },
    );
  };

  // Bulk delete selected entries
  const handleBulkDelete = async () => {
    if (selectedEntries.length === 0) {
      toast.error("No items selected");
      return;
    }

    openConfirm(
      "history-bulk-delete",
      `Delete ${selectedEntries.length} selected item${selectedEntries.length > 1 ? "s" : ""}?`,
      async () => {
        let successCount = 0;
        let failCount = 0;

        for (const id of selectedEntries) {
          const success = await deleteHistoryEntry(id);
          if (success) {
            successCount++;
          } else {
            failCount++;
          }
        }

        if (failCount === 0) {
          toast.success(
            `Successfully deleted ${successCount} item${successCount > 1 ? "s" : ""}`,
          );
        } else {
          toast.warning(
            `Deleted ${successCount} items, failed to delete ${failCount} items`,
          );
        }

        setSelectedEntries([]);
      },
    );
  };

  // Delete all entries
  const handleDeleteAll = async () => {
    if (uploadHistory.length === 0) {
      toast.error("No items to delete");
      return;
    }

    openConfirm(
      "history-delete-all",
      `Delete ALL ${uploadHistory.length} history entries? This cannot be undone.`,
      async () => {
        const success = await clearHistory();
        if (success) {
          setSelectedEntries([]);
          toast.success("All history entries deleted successfully");
        } else {
          toast.error("Failed to delete all entries");
        }
      },
    );
  };

  const handleDeleteFiltered = async () => {
    if (filteredHistory.length === 0) {
      toast.error("No matching items to delete");
      return;
    }

    const filteredIds = filteredHistory.map((entry) => entry.id);

    openConfirm(
      "history-delete-filtered",
      `Delete ALL ${filteredHistory.length} filtered item${filteredHistory.length > 1 ? "s" : ""}?`,
      async () => {
        const result = await deleteMultipleHistoryEntries(filteredIds);

        if (result.failCount === 0) {
          toast.success(
            `Successfully deleted ${result.successCount} filtered item${result.successCount > 1 ? "s" : ""}`,
          );
          setSelectedEntries([]);
          setFilterText(""); // Clear search after deletion
        } else {
          toast.warning(
            `Deleted ${result.successCount} items, failed to delete ${result.failCount} items`,
          );
        }
      },
    );
  };

  const handleSelectEntry = (e, id) => {
    e.stopPropagation();
    setSelectedEntries((prev) =>
      prev.includes(id)
        ? prev.filter((entryId) => entryId !== id)
        : [...prev, id],
    );
  };

  const handleSelectAll = () => {
    if (selectedEntries.length === filteredHistory.length) {
      setSelectedEntries([]);
    } else {
      setSelectedEntries(filteredHistory.map((entry) => entry.id));
    }
  };

  const handleClearSearch = () => {
    setFilterText("");
    setSearchField("all");
    setDateFilter("all");
  };

  const handleViewResult = (entry) => {
    if (onViewResult && entry.results && entry.results.length > 0) {
      onViewResult(entry);
    } else if (entry.status === "processing") {
      toast.error("This video is still processing. Please check back later.");
    } else if (entry.status === "failed") {
      toast.error("This video processing failed. Please try uploading again.");
    } else {
      toast.info("No detection results available for this entry.");
    }
  };

  // Loading state
  if (loading && uploadHistory.length === 0) {
    return (
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl border border-gray-700 p-12 text-center">
        <Loader2 className="h-12 w-12 text-purple-400 mx-auto mb-4 animate-spin" />
        <h3 className="text-xl font-medium text-gray-300 mb-2">
          Loading History...
        </h3>
        <p className="text-gray-500">
          Fetching your upload history from the server
        </p>
      </div>
    );
  }

  // Error state
  if (error && uploadHistory.length === 0) {
    return (
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl border border-gray-700 p-12 text-center">
        <AlertCircle className="h-12 w-12 text-red-400 mx-auto mb-4" />
        <h3 className="text-xl font-medium text-gray-300 mb-2">
          Error Loading History
        </h3>
        <p className="text-gray-500 mb-4">{error}</p>
        <button
          onClick={handleRefresh}
          className="px-4 py-2 bg-purple-500/20 text-purple-400 rounded-lg hover:bg-purple-500/30 transition-colors"
        >
          Try Again
        </button>
      </div>
    );
  }

  if (uploadHistory.length === 0) {
    return (
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl border border-gray-700 p-12 text-center">
        <div className="flex justify-center mb-4">
          <Clock className="h-16 w-16 text-gray-600" />
        </div>
        <h3 className="text-xl font-medium text-gray-300 mb-2">
          No Upload History
        </h3>
        <p className="text-gray-500">Upload videos to see your history here</p>
      </div>
    );
  }

  return (
    <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl border border-gray-700 overflow-hidden shadow-lg shadow-black/20">
      {/* Header */}
      <div className="p-4 border-b border-gray-700 bg-gradient-to-r from-gray-800/90 to-gray-900/80">
        <div className="flex items-center justify-between mb-4 flex-wrap gap-3">
          <div className="flex items-center space-x-2">
            <Clock className="h-5 w-5 text-purple-400" />
            <h2 className="text-lg font-semibold text-white">Upload History</h2>
            <span className="px-2 py-1 bg-gray-700 rounded-full text-xs text-gray-300">
              {uploadHistory.length} total
            </span>
            {filteredHistory.length !== uploadHistory.length && (
              <span className="px-2 py-1 bg-blue-500/20 text-blue-400 rounded-full text-xs">
                {filteredHistory.length} filtered
              </span>
            )}
          </div>

          <div className="relative flex items-center space-x-2 flex-wrap gap-2">
            <div className="flex items-center rounded-lg border border-gray-600 overflow-hidden">
              <button
                onClick={() => setViewMode("compact")}
                className={`px-3 py-1.5 text-xs transition-colors ${
                  viewMode === "compact"
                    ? "bg-purple-500/30 text-purple-200"
                    : "bg-gray-800/70 text-gray-300 hover:bg-gray-700"
                }`}
                title="Compact View"
              >
                Compact
              </button>
              <button
                onClick={() => setViewMode("detailed")}
                className={`px-3 py-1.5 text-xs transition-colors ${
                  viewMode === "detailed"
                    ? "bg-purple-500/30 text-purple-200"
                    : "bg-gray-800/70 text-gray-300 hover:bg-gray-700"
                }`}
                title="Detailed View"
              >
                Detailed
              </button>
            </div>

            {/* Delete All Button */}
            {uploadHistory.length > 0 && (
              <button
                onClick={handleDeleteAll}
                className="px-3 py-1.5 bg-red-500/10 text-red-400 rounded-lg hover:bg-red-500/20 text-sm flex items-center space-x-1 transition-colors"
                title="Delete All History"
              >
                <Trash2 className="h-4 w-4" />
                <span>Delete All</span>
              </button>
            )}
            {confirmState.key === "history-delete-all" && (
              <InlineConfirmPopover
                message={confirmState.message}
                onConfirm={executeConfirm}
                onCancel={closeConfirm}
              />
            )}

            {/* Delete Filtered Button */}
            {filterText && filteredHistory.length > 0 && (
              <button
                onClick={handleDeleteFiltered}
                className="px-3 py-1.5 bg-orange-500/10 text-orange-400 rounded-lg hover:bg-orange-500/20 text-sm flex items-center space-x-1 transition-colors"
                title="Delete Filtered Results"
              >
                <Search className="h-4 w-4" />
                <span>Delete Filtered ({filteredHistory.length})</span>
              </button>
            )}
            {confirmState.key === "history-delete-filtered" && (
              <InlineConfirmPopover
                message={confirmState.message}
                onConfirm={executeConfirm}
                onCancel={closeConfirm}
              />
            )}

            {/* Refresh Button */}
            <button
              onClick={handleRefresh}
              disabled={isRefreshing}
              className="px-3 py-1.5 bg-gray-700/50 text-gray-300 rounded-lg hover:bg-gray-700 text-sm flex items-center space-x-1 transition-colors"
            >
              <RefreshCw
                className={`h-4 w-4 ${isRefreshing ? "animate-spin" : ""}`}
              />
              <span>Refresh</span>
            </button>

            {/* Bulk Delete Button */}
            {selectedEntries.length > 0 && (
              <button
                onClick={handleBulkDelete}
                className="px-3 py-1.5 bg-red-500/20 text-red-400 rounded-lg hover:bg-red-500/30 text-sm flex items-center space-x-1 transition-colors"
              >
                <Trash2 className="h-4 w-4" />
                <span>Delete ({selectedEntries.length})</span>
              </button>
            )}
            {confirmState.key === "history-bulk-delete" && (
              <InlineConfirmPopover
                message={confirmState.message}
                onConfirm={executeConfirm}
                onCancel={closeConfirm}
              />
            )}
          </div>
        </div>

        {/* Advanced Filters */}
        <div className="space-y-3">
          {/* Search input */}
          <div className="flex items-center space-x-2">
            <div className="flex-1 relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-500" />
              <input
                type="text"
                placeholder={`Search by ${searchField === "all" ? "video name, prompt, or status" : searchField}...`}
                value={filterText}
                onChange={(e) => setFilterText(e.target.value)}
                className="w-full pl-10 pr-10 py-2 bg-gray-700/50 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500"
              />
              {filterText && (
                <button
                  onClick={() => setFilterText("")}
                  className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-white"
                >
                  <X className="h-4 w-4" />
                </button>
              )}
            </div>

            <select
              value={searchField}
              onChange={(e) => setSearchField(e.target.value)}
              className="px-3 py-2 bg-gray-700/50 border border-gray-600 rounded-lg text-gray-300 focus:outline-none focus:ring-2 focus:ring-purple-500"
            >
              <option value="all">All Fields</option>
              <option value="videoName">Video Name</option>
              <option value="prompt">Prompt</option>
              <option value="status">Status</option>
            </select>
          </div>

          {/* Filter row */}
          <div className="flex items-center space-x-3 flex-wrap gap-2">
            <Filter className="h-4 w-4 text-gray-500" />
            <span className="text-sm text-gray-400">Sort:</span>
            <button
              onClick={() =>
                setSortOrder(sortOrder === "desc" ? "asc" : "desc")
              }
              className="px-3 py-1.5 bg-gray-700/50 border border-gray-600 rounded-lg text-gray-300 hover:bg-gray-700 flex items-center space-x-1 text-sm transition-colors"
            >
              <RefreshCw
                className={`h-3 w-3 ${sortOrder === "asc" ? "rotate-180" : ""}`}
              />
              <span>
                {sortOrder === "desc" ? "Newest First" : "Oldest First"}
              </span>
            </button>

            <span className="text-sm text-gray-400 ml-2">Date:</span>
            <select
              value={dateFilter}
              onChange={(e) => setDateFilter(e.target.value)}
              className="px-3 py-1.5 bg-gray-700/50 border border-gray-600 rounded-lg text-gray-300 text-sm focus:outline-none focus:ring-2 focus:ring-purple-500"
            >
              <option value="all">All Time</option>
              <option value="today">Today</option>
              <option value="week">Last 7 Days</option>
              <option value="month">Last 30 Days</option>
            </select>

            {(filterText || dateFilter !== "all") && (
              <button
                onClick={handleClearSearch}
                className="px-3 py-1.5 bg-gray-600/50 text-gray-300 rounded-lg hover:bg-gray-600 text-sm transition-colors"
              >
                Clear Filters
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Select All Bar */}
      {filteredHistory.length > 0 && (
        <div className="px-4 py-2 bg-gray-700/30 border-b border-gray-700 flex items-center justify-between">
          <label className="flex items-center space-x-2 cursor-pointer">
            <input
              type="checkbox"
              checked={
                selectedEntries.length === filteredHistory.length &&
                filteredHistory.length > 0
              }
              onChange={handleSelectAll}
              className="w-4 h-4 bg-gray-700 border-gray-600 rounded focus:ring-purple-500"
            />
            <span className="text-sm text-gray-300">
              Select All ({filteredHistory.length})
            </span>
          </label>
          {selectedEntries.length > 0 && (
            <span className="text-sm text-purple-400">
              {selectedEntries.length} selected
            </span>
          )}
        </div>
      )}

      {/* History List */}
      <div className="divide-y divide-gray-700 max-h-[640px] overflow-y-auto">
        {filteredHistory.map((entry) => (
          <div
            key={entry.id}
            className="hover:bg-gray-700/30 transition-colors"
          >
            {/* Summary Row */}
            <div className="p-4">
              <div className="flex items-start space-x-4">
                {/* Checkbox */}
                <div onClick={(e) => e.stopPropagation()} className="pt-1">
                  <input
                    type="checkbox"
                    checked={selectedEntries.includes(entry.id)}
                    onChange={(e) => handleSelectEntry(e, entry.id)}
                    className="w-4 h-4 bg-gray-700 border-gray-600 rounded focus:ring-purple-500"
                  />
                </div>

                {/* Info - Click to expand */}
                <div
                  className="flex-1 cursor-pointer"
                  onClick={() => toggleExpand(entry.id)}
                >
                  <div className="flex items-center space-x-2 flex-wrap gap-2">
                    <h3 className="text-white font-medium">
                      {entry.videoName || "Untitled Video"}
                    </h3>
                    {entry.prompt && (
                      <span className="px-2 py-0.5 bg-purple-500/20 text-purple-400 rounded-full text-xs">
                        Prompt:{" "}
                        {entry.prompt.length > 30
                          ? entry.prompt.substring(0, 30) + "..."
                          : entry.prompt}
                      </span>
                    )}
                    {getStatusBadge(entry.status)}
                  </div>

                  <div className="flex items-center space-x-4 mt-2 text-sm text-gray-400 flex-wrap gap-2">
                    <span className="flex items-center">
                      <Calendar className="h-3 w-3 mr-1" />
                      {formatDate(entry.timestamp)}
                    </span>

                    {entry.results && entry.results.length > 0 && (
                      <span className="flex items-center">
                        <Eye className="h-3 w-3 mr-1" />
                        {entry.results.length} detection
                        {entry.results.length !== 1 ? "s" : ""}
                      </span>
                    )}

                    {entry.totalDetections > 0 &&
                      entry.results.length === 0 && (
                        <span className="flex items-center">
                          <Eye className="h-3 w-3 mr-1" />
                          {entry.totalDetections} detection
                          {entry.totalDetections !== 1 ? "s" : ""}
                        </span>
                      )}

                    {viewMode === "detailed" && entry.mode && (
                      <span className="px-2 py-0.5 bg-blue-500/20 text-blue-400 rounded-full text-xs">
                        {entry.mode}
                      </span>
                    )}

                    {viewMode === "detailed" && resolveVideoUrl(entry) && (
                      <span className="px-2 py-0.5 bg-indigo-500/20 text-indigo-300 rounded-full text-xs flex items-center gap-1">
                        <Video className="h-3 w-3" />
                        Video Available
                      </span>
                    )}

                    {viewMode === "detailed" &&
                      entry.processingTime &&
                      entry.processingTime !== "0" && (
                        <span className="text-xs text-gray-500 bg-gray-800/70 px-2 py-0.5 rounded-md border border-gray-700">
                          Time: {entry.processingTime}
                        </span>
                      )}
                  </div>
                </div>

                {/* Actions */}
                <div className="relative flex items-center space-x-1">
                  <button
                    onClick={() => handleViewResult(entry)}
                    disabled={
                      entry.status !== "completed" ||
                      (entry.results && entry.results.length === 0)
                    }
                    className={`p-2 rounded-lg transition-colors ${
                      entry.status === "completed" &&
                      entry.results &&
                      entry.results.length > 0
                        ? "hover:bg-blue-500/20 text-blue-400"
                        : "opacity-50 cursor-not-allowed text-gray-500"
                    }`}
                    title="View Results"
                  >
                    <Eye className="h-4 w-4" />
                  </button>

                  <button
                    onClick={(e) => handleDelete(e, entry.id, entry.videoName)}
                    className="p-2 hover:bg-red-500/20 rounded-lg text-red-400 transition-colors"
                    title="Delete"
                  >
                    <Trash2 className="h-4 w-4" />
                  </button>
                  {confirmState.key === `history-single-${entry.id}` && (
                    <InlineConfirmPopover
                      message={confirmState.message}
                      onConfirm={executeConfirm}
                      onCancel={closeConfirm}
                    />
                  )}

                  <button
                    onClick={() => toggleExpand(entry.id)}
                    className="p-2 hover:bg-gray-600 rounded-lg text-gray-400"
                  >
                    {expandedId === entry.id ? (
                      <ChevronUp className="h-4 w-4" />
                    ) : (
                      <ChevronDown className="h-4 w-4" />
                    )}
                  </button>
                </div>
              </div>
            </div>

            {/* Expanded Details - Same as before */}
            {viewMode === "detailed" && expandedId === entry.id && (
              <div className="px-4 pb-4 pl-20">
                <div className="bg-gray-900/40 border border-gray-700 rounded-lg p-4 ml-12">
                  {/* ... keep the existing expanded details content ... */}
                  {resolveVideoUrl(entry) && (
                    <div className="mb-4">
                      <p className="text-xs text-gray-500 mb-2">
                        Video Preview
                      </p>
                      <div className="bg-gray-900/60 border border-gray-600 rounded-lg p-2">
                        <video
                          controls
                          preload="metadata"
                          className="w-full max-h-80 rounded-lg bg-black"
                          src={resolveVideoUrl(entry)}
                        >
                          Your browser does not support video playback.
                        </video>
                      </div>
                    </div>
                  )}

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                    {entry.prompt && (
                      <div className="bg-gray-800/50 p-2 rounded">
                        <p className="text-xs text-gray-500 mb-1">
                          Prompt / Search Text
                        </p>
                        <p className="text-sm text-white flex items-center">
                          <FileText className="h-3 w-3 mr-1 text-green-400" />
                          {entry.prompt}
                        </p>
                      </div>
                    )}

                    {entry.processingTime && entry.processingTime !== "0" && (
                      <div className="bg-gray-800/50 p-2 rounded">
                        <p className="text-xs text-gray-500 mb-1">
                          Processing Time
                        </p>
                        <p className="text-sm font-medium text-white">
                          {entry.processingTime}
                        </p>
                      </div>
                    )}

                    {entry.status && (
                      <div className="bg-gray-800/50 p-2 rounded">
                        <p className="text-xs text-gray-500 mb-1">Status</p>
                        <p className="text-sm font-medium text-white">
                          {entry.status}
                        </p>
                      </div>
                    )}
                  </div>

                  {entry.stats && Object.keys(entry.stats).length > 0 && (
                    <div className="border-t border-gray-600 pt-4">
                      <p className="text-xs text-gray-500 mb-2">Statistics</p>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                        {entry.stats.totalDetections > 0 && (
                          <div className="bg-gray-800/50 rounded p-2">
                            <p className="text-xs text-gray-400">
                              Total Detections
                            </p>
                            <p className="text-sm font-medium text-white">
                              {entry.stats.totalDetections}
                            </p>
                          </div>
                        )}
                        {entry.stats.totalVehicles > 0 && (
                          <div className="bg-gray-800/50 rounded p-2">
                            <p className="text-xs text-gray-400">Vehicles</p>
                            <p className="text-sm font-medium text-white">
                              {entry.stats.totalVehicles}
                            </p>
                          </div>
                        )}
                        {entry.stats.totalPersons > 0 && (
                          <div className="bg-gray-800/50 rounded p-2">
                            <p className="text-xs text-gray-400">Persons</p>
                            <p className="text-sm font-medium text-white">
                              {entry.stats.totalPersons}
                            </p>
                          </div>
                        )}
                        {entry.stats.totalHelmet > 0 && (
                          <div className="bg-gray-800/50 rounded p-2">
                            <p className="text-xs text-gray-400">With Helmet</p>
                            <p className="text-sm font-medium text-white">
                              {entry.stats.totalHelmet}
                            </p>
                          </div>
                        )}
                      </div>
                    </div>
                  )}

                  {entry.results && entry.results.length > 0 && (
                    <div className="border-t border-gray-600 pt-4 mt-4">
                      <p className="text-xs text-gray-500 mb-2">
                        Detections Preview ({entry.results.length} total)
                      </p>
                      <div className="grid grid-cols-3 sm:grid-cols-5 md:grid-cols-8 gap-2">
                        {entry.results.slice(0, 8).map((result, idx) => (
                          <div key={idx} className="relative group">
                            {resolveDetectionPreviewImage(result) ? (
                              <img
                                src={resolveDetectionPreviewImage(result)}
                                alt="Detection"
                                className="w-full h-16 object-cover rounded-lg border border-gray-600"
                                onError={(e) => {
                                  e.target.onerror = null;
                                }}
                              />
                            ) : (
                              <div className="w-full h-16 bg-gray-700 rounded-lg flex items-center justify-center">
                                <Eye className="h-5 w-5 text-gray-500" />
                              </div>
                            )}
                            <div className="absolute inset-0 bg-black/75 opacity-0 group-hover:opacity-100 transition-opacity rounded-lg flex items-center justify-center">
                              <div className="text-center">
                                <span className="text-xs text-white font-medium">
                                  {result.confidencePercentage ||
                                    Math.round(result.confidence * 100) ||
                                    0}
                                  %
                                </span>
                                <br />
                                <span className="text-xs text-gray-300">
                                  {result.object || "Object"}
                                </span>
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        ))}
      </div>

      {filteredHistory.length === 0 && (
        <div className="p-8 text-center">
          <p className="text-gray-400">
            {filterText || dateFilter !== "all"
              ? "No history entries match your search criteria"
              : "No history entries available"}
          </p>
          {(filterText || dateFilter !== "all") && (
            <button
              onClick={handleClearSearch}
              className="mt-3 px-4 py-2 bg-purple-500/20 text-purple-400 rounded-lg hover:bg-purple-500/30 text-sm"
            >
              Clear Filters
            </button>
          )}
        </div>
      )}
    </div>
  );
};

export default PreviousResultsTable;
