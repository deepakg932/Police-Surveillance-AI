import React, { useState, useEffect } from "react";
import {
  Play,
  Download,
  Eye,
  Info,
  X,
  Search,
  Trash2,
  Filter,
  ChevronDown,
  ChevronUp,
  Loader2,
} from "lucide-react";
import { toast } from "react-hot-toast";
import InlineConfirmPopover from "../Common/InlineConfirmPopover";

const DetectedPersonsTable = ({
  persons,
  onDeleteSingleDetection,
  onDeleteDetections,
  onDeleteAllDetections,
  onDeleteComplete,
  onPersonsUpdate,
  apiBaseUrl,
  authFetch,
}) => {
  const [selectedPerson, setSelectedPerson] = useState(null);
  const [showModal, setShowModal] = useState(false);
  const [searchText, setSearchText] = useState("");
  const [searchField, setSearchField] = useState("all");
  const [sortField, setSortField] = useState("timestamp");
  const [sortOrder, setSortOrder] = useState("asc");
  const [filterAttributes, setFilterAttributes] = useState({
    helmet: false,
    vehicle: false,
    highConfidence: false,
    lowConfidence: false,
  });
  const [selectedDetections, setSelectedDetections] = useState([]);
  const [isDeleting, setIsDeleting] = useState(false);
  const [localPersons, setLocalPersons] = useState(persons);
  const [confirmState, setConfirmState] = useState({
    key: null,
    message: "",
    onConfirm: null,
  });

  console.log(selectedPerson)

  // Update local persons when props change
  useEffect(() => {
    setLocalPersons(persons);
  }, [persons]);

  // Map of colors to their actual classes
  const colorClasses = {
    blue: {
      bg: "bg-blue-500/20",
      text: "text-blue-400",
      border: "border-blue-500/30",
    },
    yellow: {
      bg: "bg-yellow-500/20",
      text: "text-yellow-400",
      border: "border-yellow-500/30",
    },
    green: {
      bg: "bg-green-500/20",
      text: "text-green-400",
      border: "border-green-500/30",
    },
    purple: {
      bg: "bg-purple-500/20",
      text: "text-purple-400",
      border: "border-purple-500/30",
    },
    gray: {
      bg: "bg-gray-500/20",
      text: "text-gray-400",
      border: "border-gray-500/30",
    },
    red: {
      bg: "bg-red-500/20",
      text: "text-red-400",
      border: "border-red-500/30",
    },
  };

  const getAttributeBadge = (condition, text, color) => {
    return (
      condition && (
        <span
          className={`px-2 py-0.5 ${colorClasses[color].bg} ${colorClasses[color].text} rounded text-xs border ${colorClasses[color].border}`}
        >
          {text}
        </span>
      )
    );
  };

  const handleViewThumbnail = (thumbnailUrl) => {
    if (thumbnailUrl) {
      window.open(thumbnailUrl, "_blank");
    }
  };

  const resolveDetectionImageUrl = (person) => {
    const rawPath = person?.imagepath_full ||person?.image_path || "";
console.log(person)

    if (!rawPath || typeof rawPath !== "string") {
      const directUrl =
        person?.imagepath_full || person?.image_path || "";
      return typeof directUrl === "string" ? directUrl : "";
    }
    if (/^https?:\/\//i.test(rawPath)) {
      return rawPath;
    }

    const cleanedPath = rawPath.replace(/^\/+/, "");
    return `https://workingcart.com/files/${cleanedPath}`;
  };

  const handleViewDetails = (person) => {
    setSelectedPerson(person);
    setShowModal(true);
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

  const closeModal = () => {
    setShowModal(false);
    setSelectedPerson(null);
  };

  // Update local state after deletion
  const updateLocalStateAfterDelete = (deletedTrackingIds) => {
    const deletedIds = Array.isArray(deletedTrackingIds)
      ? deletedTrackingIds
      : [deletedTrackingIds];
    setLocalPersons((prev) =>
      prev.filter((person) => !deletedIds.includes(person.trackingId)),
    );
    setSelectedDetections((prev) =>
      prev.filter((id) => !deletedIds.includes(id)),
    );
  };

  // API Delete Functions
  // const deleteSingleDetection = async (trackingId) => {
  //   if (!trackingId) {
  //     toast.error("Invalid tracking ID");
  //     return false;
  //   }

  //   setIsDeleting(true);
  //   try {
  //     const url = `${apiBaseUrl || import.meta.env.VITE_API_BASE_URL}/video/delete/${trackingId}`;
  //     const response = await authFetch(url, {
  //       method: 'DELETE',
  //     });

  //     if (!response.ok) {
  //       throw new Error(`Delete failed: ${response.status}`);
  //     }

  //     const result = await response.json();
  //     console.log("Delete response:", result);

  //     // Update local state immediately
  //     updateLocalStateAfterDelete(trackingId);

  //     toast.success(`Successfully deleted detection ${trackingId}`);

  //     // Notify parent to refresh if needed
  //     if (onDeleteComplete) {
  //       onDeleteComplete();
  //     }

  //     // Notify parent of updated persons list
  //     if (onPersonsUpdate) {
  //       onPersonsUpdate(localPersons.filter(p => p.trackingId !== trackingId));
  //     }

  //     return true;
  //   } catch (error) {
  //     console.error("Error deleting detection:", error);
  //     toast.error(`Failed to delete: ${error.message}`);
  //     return false;
  //   } finally {
  //     setIsDeleting(false);
  //   }
  // };

  const deleteMultipleDetections = async (trackingIds) => {
    if (!trackingIds || trackingIds.length === 0) {
      toast.error("No tracking IDs provided");
      return false;
    }

    setIsDeleting(true);
    let successCount = 0;
    let failCount = 0;
    const successfulDeletes = [];

    try {
      for (const trackingId of trackingIds) {
        try {
          const url = `${apiBaseUrl || import.meta.env.VITE_API_BASE_URL}/video/delete/${trackingId}`;
          const response = await authFetch(url, {
            method: "DELETE",
          });

          if (response.ok) {
            successCount++;
            successfulDeletes.push(trackingId);
          } else {
            failCount++;
            console.error(`Failed to delete ${trackingId}: ${response.status}`);
          }
        } catch (error) {
          failCount++;
          console.error(`Error deleting ${trackingId}:`, error);
        }
      }

      // Update local state with successfully deleted items
      if (successfulDeletes.length > 0) {
        updateLocalStateAfterDelete(successfulDeletes);
      }

      if (failCount === 0) {
        toast.success(
          `Successfully deleted ${successCount} detection${successCount > 1 ? "s" : ""}`,
        );

        // Notify parent to refresh
        if (onDeleteComplete) {
          onDeleteComplete();
        }

        // Notify parent of updated persons list
        if (onPersonsUpdate) {
          onPersonsUpdate(
            localPersons.filter(
              (p) => !successfulDeletes.includes(p.trackingId),
            ),
          );
        }

        return true;
      } else {
        toast.warning(
          `Deleted ${successCount} items, failed to delete ${failCount} items`,
        );
        return successCount > 0;
      }
    } catch (error) {
      console.error("Error in batch delete:", error);
      toast.error(`Delete operation failed: ${error.message}`);
      return false;
    } finally {
      setIsDeleting(false);
    }
  };

  const deleteAllDetections = async () => {
    setIsDeleting(true);
    try {
      // First, get all tracking IDs to delete
      const allTrackingIds = localPersons
        .map((p) => p.trackingId)
        .filter((id) => id);

      if (allTrackingIds.length === 0) {
        toast.error("No detections to delete");
        return false;
      }

      // Try batch delete if available, otherwise delete one by one
      let success = false;

      // Try using the deleteAll endpoint first
      const deleteAllUrl = `${apiBaseUrl || import.meta.env.VITE_API_BASE_URL}/video/deleteall`;

      try {
        const response = await authFetch(deleteAllUrl, {
          method: "DELETE",
        });

        if (response.ok) {
          const result = await response.json();
          console.log("Delete all response:", result);
          success = true;
        } else {
          // If deleteAll fails, delete one by one
          console.log("Delete all failed, deleting individually...");
          let successCount = 0;
          for (const trackingId of allTrackingIds) {
            const deleteUrl = `${apiBaseUrl || import.meta.env.VITE_API_BASE_URL}/video/delete/${trackingId}`;
            const deleteResponse = await authFetch(deleteUrl, {
              method: "DELETE",
            });
            if (deleteResponse.ok) {
              successCount++;
            }
          }
          success = successCount > 0;
        }
      } catch (error) {
        // If batch delete fails, delete one by one
        console.log("Batch delete error, deleting individually:", error);
        let successCount = 0;
        for (const trackingId of allTrackingIds) {
          const deleteUrl = `${apiBaseUrl || import.meta.env.VITE_API_BASE_URL}/video/delete/${trackingId}`;
          const deleteResponse = await authFetch(deleteUrl, {
            method: "DELETE",
          });
          if (deleteResponse.ok) {
            successCount++;
          }
        }
        success = successCount > 0;
      }

      if (success) {
        // Clear all local persons
        setLocalPersons([]);
        setSelectedDetections([]);

        toast.success(
          `Successfully deleted all ${allTrackingIds.length} detections`,
        );

        // Notify parent to refresh and update stats
        if (onDeleteComplete) {
          onDeleteComplete();
        }

        // Notify parent of empty persons list to update total detections
        if (onPersonsUpdate) {
          onPersonsUpdate([]);
        }

        return true;
      } else {
        toast.error("Failed to delete all detections");
        return false;
      }
    } catch (error) {
      console.error("Error deleting all detections:", error);
      toast.error(`Failed to delete all: ${error.message}`);
      return false;
    } finally {
      setIsDeleting(false);
    }
  };

  // Filter and search logic (using localPersons)
  const filteredPersons = localPersons.filter((person) => {
    // Text search
    if (searchText) {
      const searchLower = searchText.toLowerCase();
      switch (searchField) {
        case "trackingId":
          return person.trackingId
            ?.toString()
            .toLowerCase()
            .includes(searchLower);
        case "object":
          return person.object?.toLowerCase().includes(searchLower);
        case "timestamp":
          return person.timestamp?.toString().includes(searchLower);
        default:
          return (
            person.trackingId?.toString().toLowerCase().includes(searchLower) ||
            person.object?.toLowerCase().includes(searchLower) ||
            person.timestamp?.toString().includes(searchLower) ||
            person.confidence?.toString().includes(searchLower)
          );
      }
    }

    // Attribute filters
    if (filterAttributes.helmet && !person.object?.includes("helmet")) {
      return false;
    }
    if (
      filterAttributes.vehicle &&
      !(
        person.object?.includes("motorcycle") ||
        person.object?.includes("bike") ||
        person.object?.includes("car")
      )
    ) {
      return false;
    }
    if (
      filterAttributes.highConfidence &&
      (!person.confidence || person.confidence < 0.8)
    ) {
      return false;
    }
    if (
      filterAttributes.lowConfidence &&
      (!person.confidence || person.confidence > 0.6)
    ) {
      return false;
    }

    return true;
  });

  // Sort logic
  const sortedPersons = [...filteredPersons].sort((a, b) => {
    let aVal = a[sortField];
    let bVal = b[sortField];

    if (sortField === "confidence") {
      aVal = a.confidence || 0;
      bVal = b.confidence || 0;
    } else if (sortField === "timestamp") {
      aVal = parseFloat(a.timestamp || 0);
      bVal = parseFloat(b.timestamp || 0);
    }

    if (aVal < bVal) return sortOrder === "asc" ? -1 : 1;
    if (aVal > bVal) return sortOrder === "asc" ? 1 : -1;
    return 0;
  });

  // Handle sort click
  const handleSort = (field) => {
    if (sortField === field) {
      setSortOrder(sortOrder === "asc" ? "desc" : "asc");
    } else {
      setSortField(field);
      setSortOrder("asc");
    }
  };

  // Get sort icon
  const getSortIcon = (field) => {
    if (sortField !== field) return null;
    return sortOrder === "asc" ? (
      <ChevronUp className="h-3 w-3" />
    ) : (
      <ChevronDown className="h-3 w-3" />
    );
  };

  // Selection handlers
  const handleSelectDetection = (e, trackingId) => {
    e.stopPropagation();
    setSelectedDetections((prev) =>
      prev.includes(trackingId)
        ? prev.filter((id) => id !== trackingId)
        : [...prev, trackingId],
    );
  };

  const handleSelectAll = () => {
    if (selectedDetections.length === sortedPersons.length) {
      setSelectedDetections([]);
    } else {
      setSelectedDetections(
        sortedPersons.map((p) => p.trackingId).filter((id) => id),
      );
    }
  };

  // Delete handlers
  const handleDeleteSingle = async (trackingId, objectName) => {
    openConfirm(
      `single-${trackingId}`,
      `Delete detection ${trackingId} (${objectName || "Unknown"})?`,
      async () => {
      if (onDeleteSingleDetection) {
        const success = await onDeleteSingleDetection(trackingId);
        if (success) {
          // Remove from local selection if needed
          setSelectedDetections((prev) =>
            prev.filter((id) => id !== trackingId),
          );
        }
      }
      },
    );
  };

  const handleBulkDelete = async () => {
    if (selectedDetections.length === 0) {
      toast.error("No items selected");
      return;
    }

    openConfirm(
      "bulk-delete",
      `Delete ${selectedDetections.length} selected detection${selectedDetections.length > 1 ? "s" : ""}?`,
      async () => {
      if (onDeleteDetections) {
        const success = await onDeleteDetections(selectedDetections);
        if (success) {
          setSelectedDetections([]);
        }
      }
      },
    );
  };

  const handleDeleteFiltered = async () => {
    if (sortedPersons.length === 0) {
      toast.error("No matching items to delete");
      return;
    }

    const filteredTrackingIds = sortedPersons
      .map((p) => p.trackingId)
      .filter((id) => id);

    openConfirm(
      "filtered-delete",
      `Delete ALL ${sortedPersons.length} filtered detection${sortedPersons.length > 1 ? "s" : ""}?`,
      async () => {
      await deleteMultipleDetections(filteredTrackingIds);
      setSearchText("");
      setFilterAttributes({
        helmet: false,
        vehicle: false,
        highConfidence: false,
        lowConfidence: false,
      });
      },
    );
  };

  const handleDeleteAll = async () => {
    if (localPersons.length === 0) {
      toast.error("No detections to delete");
      return;
    }

    openConfirm(
      "delete-all",
      `Delete ALL ${localPersons.length} detections? This cannot be undone.`,
      async () => {
      if (onDeleteAllDetections) {
        // Use the parent's deleteAll handler
        const success = await onDeleteAllDetections();
        if (success) {
          setLocalPersons([]);
          setSelectedDetections([]);
        }
      } else {
        // Fallback to local deleteAll if parent handler not provided
        await deleteAllDetections();
      }
      },
    );
  };

  const handleClearFilters = () => {
    setSearchText("");
    setSearchField("all");
    setFilterAttributes({
      helmet: false,
      vehicle: false,
      highConfidence: false,
      lowConfidence: false,
    });
    setSortField("timestamp");
    setSortOrder("asc");
  };

  if (!localPersons || localPersons.length === 0) {
    return (
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl border border-gray-700 p-8 text-center">
        <p className="text-gray-400">No detection data available</p>
      </div>
    );
  }

  return (
    <>
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl border border-gray-700">
        {/* Header with Controls */}
        <div className="p-4 border-b border-gray-700 space-y-3">
          <div className="flex items-center justify-between flex-wrap gap-3">
            <div className="flex items-center space-x-2">
              <h3 className="text-lg font-semibold text-white">
                Detected Objects
              </h3>
              <span className="px-2 py-1 bg-gray-700 rounded-full text-xs text-gray-300">
                {localPersons.length} total
              </span>
              {sortedPersons.length !== localPersons.length && (
                <span className="px-2 py-1 bg-blue-500/20 text-blue-400 rounded-full text-xs">
                  {sortedPersons.length} filtered
                </span>
              )}
            </div>

            <div className="relative flex items-center space-x-2 flex-wrap gap-2">
              {/* Delete All Button */}
              {localPersons.length > 0 && (
                <button
                  onClick={handleDeleteAll}
                  disabled={isDeleting}
                  className="px-3 py-1.5 bg-red-500/10 text-red-400 rounded-lg hover:bg-red-500/20 text-sm flex items-center space-x-1 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isDeleting ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <Trash2 className="h-4 w-4" />
                  )}
                  <span>Delete All</span>
                </button>
              )}
              {confirmState.key === "delete-all" && (
                <InlineConfirmPopover
                  message={confirmState.message}
                  onConfirm={executeConfirm}
                  onCancel={closeConfirm}
                />
              )}

              {/* Bulk Delete Button */}
              {selectedDetections.length > 0 && (
                <button
                  onClick={handleBulkDelete}
                  disabled={isDeleting}
                  className="px-3 py-1.5 bg-red-500/20 text-red-400 rounded-lg hover:bg-red-500/30 text-sm flex items-center space-x-1 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isDeleting ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <Trash2 className="h-4 w-4" />
                  )}
                  <span>Delete ({selectedDetections.length})</span>
                </button>
              )}
              {confirmState.key === "bulk-delete" && (
                <InlineConfirmPopover
                  message={confirmState.message}
                  onConfirm={executeConfirm}
                  onCancel={closeConfirm}
                />
              )}

              {/* Delete Filtered Button */}
              {(searchText || Object.values(filterAttributes).some((v) => v)) &&
                sortedPersons.length > 0 && (
                  <button
                    onClick={handleDeleteFiltered}
                    disabled={isDeleting}
                    className="px-3 py-1.5 bg-orange-500/20 text-orange-400 rounded-lg hover:bg-orange-500/30 text-sm flex items-center space-x-1 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {isDeleting ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <Search className="h-4 w-4" />
                    )}
                    <span>Delete Filtered ({sortedPersons.length})</span>
                  </button>
                )}
              {confirmState.key === "filtered-delete" && (
                <InlineConfirmPopover
                  message={confirmState.message}
                  onConfirm={executeConfirm}
                  onCancel={closeConfirm}
                />
              )}
            </div>
          </div>

          {/* Search and Filters - Same as before */}
          <div className="space-y-3">
            <div className="flex items-center space-x-2">
              <div className="flex-1 relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-500" />
                <input
                  type="text"
                  placeholder={`Search by ${searchField === "all" ? "tracking ID, object, or timestamp" : searchField}...`}
                  value={searchText}
                  onChange={(e) => setSearchText(e.target.value)}
                  className="w-full pl-10 pr-4 py-2 bg-gray-700/50 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500"
                />
              </div>

              <select
                value={searchField}
                onChange={(e) => setSearchField(e.target.value)}
                className="px-3 py-2 bg-gray-700/50 border border-gray-600 rounded-lg text-gray-300 focus:outline-none focus:ring-2 focus:ring-purple-500"
              >
                <option value="all">All Fields</option>
                <option value="trackingId">Tracking ID</option>
                <option value="object">Object Type</option>
                <option value="timestamp">Timestamp</option>
              </select>
            </div>

            <div className="flex items-center space-x-2 flex-wrap gap-2">
              <Filter className="h-4 w-4 text-gray-500" />
              <span className="text-sm text-gray-400">Filters:</span>

              <button
                onClick={() =>
                  setFilterAttributes((prev) => ({
                    ...prev,
                    helmet: !prev.helmet,
                  }))
                }
                className={`px-2 py-1 rounded text-xs transition-colors ${
                  filterAttributes.helmet
                    ? "bg-yellow-500/30 text-yellow-400 border border-yellow-500/50"
                    : "bg-gray-700/50 text-gray-400 hover:bg-gray-700"
                }`}
              >
                Helmet Only
              </button>

              <button
                onClick={() =>
                  setFilterAttributes((prev) => ({
                    ...prev,
                    vehicle: !prev.vehicle,
                  }))
                }
                className={`px-2 py-1 rounded text-xs transition-colors ${
                  filterAttributes.vehicle
                    ? "bg-green-500/30 text-green-400 border border-green-500/50"
                    : "bg-gray-700/50 text-gray-400 hover:bg-gray-700"
                }`}
              >
                Vehicle Only
              </button>

              <button
                onClick={() =>
                  setFilterAttributes((prev) => ({
                    ...prev,
                    highConfidence: !prev.highConfidence,
                  }))
                }
                className={`px-2 py-1 rounded text-xs transition-colors ${
                  filterAttributes.highConfidence
                    ? "bg-green-500/30 text-green-400 border border-green-500/50"
                    : "bg-gray-700/50 text-gray-400 hover:bg-gray-700"
                }`}
              >
                High Confidence (≥80%)
              </button>

              <button
                onClick={() =>
                  setFilterAttributes((prev) => ({
                    ...prev,
                    lowConfidence: !prev.lowConfidence,
                  }))
                }
                className={`px-2 py-1 rounded text-xs transition-colors ${
                  filterAttributes.lowConfidence
                    ? "bg-yellow-500/30 text-yellow-400 border border-yellow-500/50"
                    : "bg-gray-700/50 text-gray-400 hover:bg-gray-700"
                }`}
              >
                Low Confidence (≤60%)
              </button>

              {(searchText ||
                Object.values(filterAttributes).some((v) => v) ||
                sortField !== "timestamp" ||
                sortOrder !== "asc") && (
                <button
                  onClick={handleClearFilters}
                  className="px-2 py-1 bg-gray-600/50 text-gray-300 rounded hover:bg-gray-600 text-xs transition-colors"
                >
                  Clear All
                </button>
              )}
            </div>
          </div>

          {/* Select All Bar */}
          {sortedPersons.length > 0 && (
            <div className="flex items-center justify-between pt-2 border-t border-gray-700">
              <label className="flex items-center space-x-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={
                    selectedDetections.length === sortedPersons.length &&
                    sortedPersons.length > 0
                  }
                  onChange={handleSelectAll}
                  disabled={isDeleting}
                  className="w-4 h-4 bg-gray-700 border-gray-600 rounded focus:ring-purple-500 disabled:opacity-50"
                />
                <span className="text-sm text-gray-300">
                  Select All ({sortedPersons.length})
                </span>
              </label>
              {selectedDetections.length > 0 && (
                <span className="text-sm text-purple-400">
                  {selectedDetections.length} selected
                </span>
              )}
            </div>
          )}
        </div>

        {/* Table - Same as before but using sortedPersons */}
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-700/50">
              <tr>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider w-8">
                  <input
                    type="checkbox"
                    checked={
                      selectedDetections.length === sortedPersons.length &&
                      sortedPersons.length > 0
                    }
                    onChange={handleSelectAll}
                    disabled={isDeleting}
                    className="w-4 h-4 bg-gray-700 border-gray-600 rounded focus:ring-purple-500 disabled:opacity-50"
                  />
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                  <button
                    onClick={() => handleSort("trackingId")}
                    className="flex items-center space-x-1 hover:text-white"
                  >
                    <span>Tracking ID</span>
                    {getSortIcon("trackingId")}
                  </button>
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                  Image
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                  <button
                    onClick={() => handleSort("timestamp")}
                    className="flex items-center space-x-1 hover:text-white"
                  >
                    <span>Timestamp (s)</span>
                    {getSortIcon("timestamp")}
                  </button>
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                  <button
                    onClick={() => handleSort("object")}
                    className="flex items-center space-x-1 hover:text-white"
                  >
                    <span>Object Type</span>
                    {getSortIcon("object")}
                  </button>
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                  Attributes
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                  <button
                    onClick={() => handleSort("confidence")}
                    className="flex items-center space-x-1 hover:text-white"
                  >
                    <span>Confidence</span>
                    {getSortIcon("confidence")}
                  </button>
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-700">
              {sortedPersons.map((person) => (
                <tr
                  key={person.trackingId}
                  className="hover:bg-gray-700/30 transition-colors"
                >
                  <td className="px-4 py-3">
                    <input
                      type="checkbox"
                      checked={selectedDetections.includes(person.trackingId)}
                      onChange={(e) =>
                        handleSelectDetection(e, person.trackingId)
                      }
                      disabled={isDeleting}
                      className="w-4 h-4 bg-gray-700 border-gray-600 rounded focus:ring-purple-500 disabled:opacity-50"
                    />
                  </td>
                  <td className="px-4 py-3 text-white font-mono text-sm">
                    <div className="space-y-1">
                      <div>{person.trackingId}</div>
                      {person.commonPersonId && (
                        <span className="inline-block px-2 py-0.5 text-[10px] rounded bg-blue-500/20 text-blue-300 border border-blue-500/30">
                          {person.commonPersonId}
                        </span>
                      )}
                    </div>
                  </td>
                  <td className="px-4 py-3">
                    {resolveDetectionImageUrl(person) ? (
                      <img
                        src={`${resolveDetectionImageUrl(person)}`}
                        alt={`Detection ${person.trackingId}`}
                        className="h-16 w-16 object-cover rounded-lg border border-gray-700 cursor-pointer hover:opacity-80 transition-opacity"
                        onClick={() => handleViewDetails(person)}
                        onError={(e) => {
                          e.target.onerror = null;
                          e.target.src = "/placeholder-image.png";
                        }}
                      />
                    ) : (
                      <div
                        className="h-16 w-16 bg-gray-700 rounded-lg flex items-center justify-center cursor-pointer hover:bg-gray-600 transition-colors"
                        onClick={() => handleViewDetails(person)}
                      >
                        <span className="text-xs text-gray-400">No image</span>
                      </div>
                    )}
                  </td>
                  <td className="px-4 py-3 text-white">
                    <span className="text-sm font-medium">
                      {person.timestamp}s
                    </span>
                  </td>
                  <td className="px-4 py-3">
                    <span className="text-white text-sm">
                      {person.object || "Unknown"}
                    </span>
                  </td>
                  <td className="px-4 py-3">
                    <div className="flex flex-wrap gap-1">
                      {getAttributeBadge(
                        person.object?.includes("helmet"),
                        "Helmet",
                        "yellow",
                      )}
                      {getAttributeBadge(
                        person.object?.includes("motorcycle") ||
                          person.object?.includes("bike"),
                        "Motorcycle",
                        "green",
                      )}
                      {getAttributeBadge(
                        person.object?.includes("car"),
                        "Car",
                        "blue",
                      )}
                      {getAttributeBadge(
                        person.object?.includes("pink"),
                        "Pink",
                        "purple",
                      )}
                      {!person.object?.includes("helmet") &&
                        !person.object?.includes("motorcycle") &&
                        !person.object?.includes("bike") &&
                        !person.object?.includes("car") &&
                        !person.object?.includes("pink") && (
                          <span className="text-xs text-gray-500">
                            No attributes
                          </span>
                        )}
                    </div>
                  </td>
                  <td className="px-4 py-3">
                    <div className="flex items-center">
                      <span
                        className={`px-2 py-1 rounded text-xs font-medium ${
                          person.confidence >= 0.8
                            ? "bg-green-500/20 text-green-400"
                            : person.confidence >= 0.6
                              ? "bg-yellow-500/20 text-yellow-400"
                              : "bg-red-500/20 text-red-400"
                        }`}
                      >
                        {Math.round(person.confidence * 100)}%
                      </span>
                    </div>
                  </td>
                  <td className="px-4 py-3">
                    <div className="relative flex items-center space-x-1">
                      <button
                        onClick={() => handleViewDetails(person)}
                        disabled={isDeleting}
                        className="p-2 hover:bg-blue-600/20 rounded-lg transition-colors group disabled:opacity-50"
                        title="View Details"
                      >
                        <Eye className="h-4 w-4 text-blue-400 group-hover:text-blue-300" />
                      </button>
                      <button
                        onClick={() =>
                          handleDeleteSingle(person.trackingId, person.object)
                        }
                        disabled={isDeleting}
                        className="p-2 hover:bg-red-500/20 rounded-lg transition-colors group disabled:opacity-50"
                        title="Delete"
                      >
                        {isDeleting ? (
                          <Loader2 className="h-4 w-4 text-red-400 animate-spin" />
                        ) : (
                          <Trash2 className="h-4 w-4 text-red-400 group-hover:text-red-300" />
                        )}
                      </button>
                      {confirmState.key === `single-${person.trackingId}` && (
                        <InlineConfirmPopover
                          message={confirmState.message}
                          onConfirm={executeConfirm}
                          onCancel={closeConfirm}
                        />
                      )}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* No results message */}
        {sortedPersons.length === 0 && searchText && (
          <div className="p-8 text-center">
            <p className="text-gray-400">
              No detections match your search criteria
            </p>
            <button
              onClick={handleClearFilters}
              className="mt-3 px-4 py-2 bg-purple-500/20 text-purple-400 rounded-lg hover:bg-purple-500/30 text-sm"
            >
              Clear Filters
            </button>
          </div>
        )}
      </div>

      {/* Details Modal - Keep as is */}
      {showModal && selectedPerson && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm">
          <div className="bg-gray-800 rounded-xl border border-gray-700 max-w-2xl w-full max-h-[90vh] overflow-y-auto">
            <div className="p-4 border-b border-gray-700 flex justify-between items-center sticky top-0 bg-gray-800">
              <h3 className="text-lg font-semibold text-white">
                Detection Details - {selectedPerson.trackingId}
              </h3>
              <button
                onClick={closeModal}
                className="p-2 hover:bg-gray-700 rounded-lg transition-colors"
              >
                <X className="h-5 w-5 text-gray-400" />
              </button>
            </div>

            <div className="p-6 space-y-6">
              {/* Full Size Image */}
              <div className="space-y-2">
                <h4 className="text-sm font-medium text-gray-400">Image</h4>
                <div className="bg-gray-900 rounded-lg border border-gray-700 p-2">
                  {resolveDetectionImageUrl(selectedPerson) ? (
                    <img
                      src={resolveDetectionImageUrl(selectedPerson)}
                      alt={`Detection ${selectedPerson.trackingId}`}
                      className="w-full h-auto max-h-96 object-contain rounded-lg"
                      onError={(e) => {
                        console.error("Image failed to load:", e.target.src);
                        e.target.src = "/placeholder-image.png";
                      }}
                    />
                  ) : (
                    <div className="h-64 bg-gray-800 rounded-lg flex items-center justify-center">
                      <span className="text-gray-400">No image available</span>
                    </div>
                  )}
                </div>
              </div>

              {/* Details Grid */}
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1">
                  <p className="text-xs text-gray-500">Tracking ID</p>
                  <p className="text-sm font-medium text-white font-mono">
                    {selectedPerson.trackingId}
                  </p>
                </div>
                <div className="space-y-1">
                  <p className="text-xs text-gray-500">Timestamp</p>
                  <p className="text-sm font-medium text-white">
                    {selectedPerson.timestamp}s
                  </p>
                </div>
                <div className="space-y-1">
                  <p className="text-xs text-gray-500">Object Type</p>
                  <p className="text-sm font-medium text-white">
                    {selectedPerson.object || "Unknown"}
                  </p>
                </div>
                <div className="space-y-1">
                  <p className="text-xs text-gray-500">Confidence</p>
                  <span
                    className={`inline-block px-2 py-1 rounded text-xs font-medium ${
                      selectedPerson.confidence >= 0.8
                        ? "bg-green-500/20 text-green-400"
                        : selectedPerson.confidence >= 0.6
                          ? "bg-yellow-500/20 text-yellow-400"
                          : "bg-red-500/20 text-red-400"
                    }`}
                  >
                    {Math.round(selectedPerson.confidence * 100)}%
                  </span>
                </div>
                {/* <div className="space-y-1">
                  <p className="text-xs text-gray-500">OCR Text</p>
                  <p className="text-sm font-medium text-white">
                    {selectedPerson.ocrText || "N/A"}
                  </p>
                </div> */}
                <div className="space-y-1">
                  <p className="text-xs text-gray-500">Processing Time</p>
                  <p className="text-sm font-medium text-white">
                    {selectedPerson.processingTime || selectedPerson.processing_time || "0"}
                  </p>
                </div>
              </div>

              {/* Attributes */}
              <div className="space-y-2">
                <h4 className="text-sm font-medium text-gray-400">
                  Attributes
                </h4>
                <div className="flex flex-wrap gap-2">
                  {getAttributeBadge(
                    selectedPerson.object?.includes("helmet"),
                    "Helmet",
                    "yellow",
                  )}
                  {getAttributeBadge(
                    selectedPerson.object?.includes("motorcycle") ||
                      selectedPerson.object?.includes("bike"),
                    "Motorcycle",
                    "green",
                  )}
                  {getAttributeBadge(
                    selectedPerson.object?.includes("car"),
                    "Car",
                    "blue",
                  )}
                  {getAttributeBadge(
                    selectedPerson.object?.includes("pink"),
                    "Pink",
                    "purple",
                  )}
                  {!selectedPerson.object?.includes("helmet") &&
                    !selectedPerson.object?.includes("motorcycle") &&
                    !selectedPerson.object?.includes("bike") &&
                    !selectedPerson.object?.includes("car") &&
                    !selectedPerson.object?.includes("pink") && (
                      <span className="text-xs text-gray-500">
                        No attributes
                      </span>
                    )}
                </div>
              </div>

              {/* Action Buttons */}
              <div className="flex justify-end space-x-3 pt-4 border-t border-gray-700">
                {resolveDetectionImageUrl(selectedPerson) && (
                  <button
                    onClick={() => handleViewThumbnail(resolveDetectionImageUrl(selectedPerson))}
                    className="px-4 py-2 bg-blue-500/20 text-blue-400 rounded-lg hover:bg-blue-500/30 transition-colors flex items-center space-x-2"
                  >
                    <Eye className="h-4 w-4" />
                    <span>Open Image</span>
                  </button>
                )}
                <button
                  onClick={closeModal}
                  className="px-4 py-2 bg-gray-700 text-white rounded-lg hover:bg-gray-600 transition-colors"
                >
                  Close
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
};

export default DetectedPersonsTable;
