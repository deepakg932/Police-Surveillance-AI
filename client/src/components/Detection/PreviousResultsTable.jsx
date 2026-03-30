import React, { useState } from 'react';
import { 
  Clock, Video, FileText, Image, 
  ChevronDown, ChevronUp, Trash2, 
  Eye, Calendar, Filter, RefreshCw,
  Download, Play
} from 'lucide-react';
import { useHistory } from '../contexts/HistoryContext';

const PreviousResultsTable = ({ onViewResult }) => {
  const { uploadHistory, deleteHistoryEntry, clearHistory } = useHistory();
  const [expandedId, setExpandedId] = useState(null);
  const [filterText, setFilterText] = useState('');
  const [sortOrder, setSortOrder] = useState('desc'); // 'asc' or 'desc'
  const [selectedEntries, setSelectedEntries] = useState([]);

  // Format date
  const formatDate = (timestamp) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins} minute${diffMins > 1 ? 's' : ''} ago`;
    if (diffHours < 24) return `${diffHours} hour${diffHours > 1 ? 's' : ''} ago`;
    if (diffDays < 7) return `${diffDays} day${diffDays > 1 ? 's' : ''} ago`;
    
    return date.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  // Format file size
  const formatFileSize = (bytes) => {
    if (!bytes) return 'N/A';
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  };

  // Filter and sort history
  const filteredHistory = uploadHistory
    .filter(entry => {
      if (!filterText) return true;
      const searchText = filterText.toLowerCase();
      return (
        entry.videoName?.toLowerCase().includes(searchText) ||
        entry.prompt?.toLowerCase().includes(searchText) ||
        entry.text?.toLowerCase().includes(searchText) ||
        entry.mode?.toLowerCase().includes(searchText)
      );
    })
    .sort((a, b) => {
      const dateA = new Date(a.timestamp);
      const dateB = new Date(b.timestamp);
      return sortOrder === 'desc' ? dateB - dateA : dateA - dateB;
    });

  const toggleExpand = (id) => {
    setExpandedId(expandedId === id ? null : id);
  };

  const handleDelete = (e, id) => {
    e.stopPropagation();
    if (window.confirm('Are you sure you want to delete this history item?')) {
      deleteHistoryEntry(id);
    }
  };

  const handleViewResult = (entry) => {
    if (onViewResult) {
      onViewResult(entry);
    }
  };

  const handleSelectEntry = (e, id) => {
    e.stopPropagation();
    setSelectedEntries(prev => 
      prev.includes(id) 
        ? prev.filter(entryId => entryId !== id)
        : [...prev, id]
    );
  };

  const handleSelectAll = (e) => {
    if (e.target.checked) {
      setSelectedEntries(filteredHistory.map(entry => entry.id));
    } else {
      setSelectedEntries([]);
    }
  };

  const handleBulkDelete = () => {
    if (selectedEntries.length === 0) return;
    
    if (window.confirm(`Are you sure you want to delete ${selectedEntries.length} selected items?`)) {
      selectedEntries.forEach(id => deleteHistoryEntry(id));
      setSelectedEntries([]);
    }
  };

  if (uploadHistory.length === 0) {
    return (
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl border border-gray-700 p-12 text-center">
        <div className="flex justify-center mb-4">
          <Clock className="h-16 w-16 text-gray-600" />
        </div>
        <h3 className="text-xl font-medium text-gray-300 mb-2">No Upload History</h3>
        <p className="text-gray-500">Upload videos to see your history here</p>
      </div>
    );
  }

  return (
    <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl border border-gray-700 overflow-hidden">
      {/* Header */}
      <div className="p-4 border-b border-gray-700">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-2">
            <Clock className="h-5 w-5 text-purple-400" />
            <h2 className="text-lg font-semibold text-white">Previous Uploads History</h2>
            <span className="px-2 py-1 bg-gray-700 rounded-full text-xs text-gray-300">
              {uploadHistory.length} total
            </span>
          </div>
          
          {selectedEntries.length > 0 && (
            <button
              onClick={handleBulkDelete}
              className="px-3 py-1.5 bg-red-500/10 text-red-400 rounded-lg hover:bg-red-500/20 text-sm flex items-center space-x-1"
            >
              <Trash2 className="h-4 w-4" />
              <span>Delete Selected ({selectedEntries.length})</span>
            </button>
          )}
        </div>

        {/* Filters */}
        <div className="flex items-center space-x-3">
          <div className="flex-1 relative">
            <Filter className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-500" />
            <input
              type="text"
              placeholder="Search by video name, prompt, or mode..."
              value={filterText}
              onChange={(e) => setFilterText(e.target.value)}
              className="w-full pl-10 pr-4 py-2 bg-gray-700/50 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500"
            />
          </div>
          <button
            onClick={() => setSortOrder(sortOrder === 'desc' ? 'asc' : 'desc')}
            className="px-3 py-2 bg-gray-700/50 border border-gray-600 rounded-lg text-gray-300 hover:bg-gray-700 flex items-center space-x-1"
          >
            <RefreshCw className={`h-4 w-4 ${sortOrder === 'asc' ? 'rotate-180' : ''}`} />
            <span className="text-sm">{sortOrder === 'desc' ? 'Newest First' : 'Oldest First'}</span>
          </button>
        </div>
      </div>

      {/* History List */}
      <div className="divide-y divide-gray-700">
        {filteredHistory.map((entry) => (
          <div key={entry.id} className="hover:bg-gray-700/30 transition-colors">
            {/* Summary Row */}
            <div className="p-4">
              <div className="flex items-center space-x-4">
                {/* Checkbox */}
                <div onClick={(e) => e.stopPropagation()}>
                  <input
                    type="checkbox"
                    checked={selectedEntries.includes(entry.id)}
                    onChange={(e) => handleSelectEntry(e, entry.id)}
                    className="w-4 h-4 bg-gray-700 border-gray-600 rounded focus:ring-purple-500"
                  />
                </div>

                {/* Thumbnail/Icon */}
                <div 
                  className="flex-shrink-0 cursor-pointer"
                  onClick={() => toggleExpand(entry.id)}
                >
                  {entry.thumbnail ? (
                    <img 
                      src={entry.thumbnail} 
                      alt="Thumbnail"
                      className="h-14 w-14 rounded-lg object-cover border border-gray-600"
                    />
                  ) : (
                    <div className="h-14 w-14 bg-gray-700 rounded-lg flex items-center justify-center border border-gray-600">
                      <Video className="h-7 w-7 text-gray-500" />
                    </div>
                  )}
                </div>

                {/* Info - Click to expand */}
                <div 
                  className="flex-1 cursor-pointer"
                  onClick={() => toggleExpand(entry.id)}
                >
                  <div className="flex items-center space-x-2">
                    <h3 className="text-white font-medium">
                      {entry.videoName || 'Untitled Video'}
                    </h3>
                    {entry.prompt && (
                      <span className="px-2 py-0.5 bg-purple-500/20 text-purple-400 rounded-full text-xs">
                        Prompt: {entry.prompt.length > 20 ? entry.prompt.substring(0, 20) + '...' : entry.prompt}
                      </span>
                    )}
                  </div>
                  
                  <div className="flex items-center space-x-4 mt-1 text-sm text-gray-400">
                    <span className="flex items-center">
                      <Calendar className="h-3 w-3 mr-1" />
                      {formatDate(entry.timestamp)}
                    </span>
                    
                    {entry.results && (
                      <span className="flex items-center">
                        <Eye className="h-3 w-3 mr-1" />
                        {entry.results.length} detection{entry.results.length !== 1 ? 's' : ''}
                      </span>
                    )}
                    
                    {entry.mode && (
                      <span className="px-2 py-0.5 bg-blue-500/20 text-blue-400 rounded-full text-xs">
                        {entry.mode}
                      </span>
                    )}
                  </div>
                </div>

                {/* Actions */}
                <div className="flex items-center space-x-1">
                  <button
                    onClick={() => handleViewResult(entry)}
                    className="p-2 hover:bg-blue-500/20 rounded-lg text-blue-400 transition-colors"
                    title="View Results"
                  >
                    <Eye className="h-4 w-4" />
                  </button>
                  
                  <button
                    onClick={(e) => handleDelete(e, entry.id)}
                    className="p-2 hover:bg-red-500/20 rounded-lg text-red-400 transition-colors"
                    title="Delete"
                  >
                    <Trash2 className="h-4 w-4" />
                  </button>
                  
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

            {/* Expanded Details */}
            {expandedId === entry.id && (
              <div className="px-4 pb-4 pl-20">
                <div className="bg-gray-700/30 rounded-lg p-4 ml-12">
                  {/* Input Details */}
                  <div className="grid grid-cols-2 gap-4 mb-4">
                    {entry.videoFile && (
                      <div className="bg-gray-800/50 p-2 rounded">
                        <p className="text-xs text-gray-500 mb-1">Video</p>
                        <p className="text-sm text-white flex items-center">
                          <Video className="h-3 w-3 mr-1 text-blue-400" />
                          {entry.videoFile.name}
                        </p>
                        <p className="text-xs text-gray-500 mt-1">
                          Size: {formatFileSize(entry.videoFile.size)}
                        </p>
                      </div>
                    )}
                    
                    {entry.prompt && (
                      <div className="bg-gray-800/50 p-2 rounded">
                        <p className="text-xs text-gray-500 mb-1">Prompt</p>
                        <p className="text-sm text-white flex items-center">
                          <FileText className="h-3 w-3 mr-1 text-green-400" />
                          {entry.prompt}
                        </p>
                      </div>
                    )}
                    
                    {entry.image && (
                      <div className="bg-gray-800/50 p-2 rounded">
                        <p className="text-xs text-gray-500 mb-1">Reference Image</p>
                        <p className="text-sm text-white flex items-center">
                          <Image className="h-3 w-3 mr-1 text-yellow-400" />
                          {entry.image.name}
                        </p>
                      </div>
                    )}
                    
                    {entry.text && (
                      <div className="bg-gray-800/50 p-2 rounded">
                        <p className="text-xs text-gray-500 mb-1">Text Input</p>
                        <p className="text-sm text-white flex items-center">
                          <FileText className="h-3 w-3 mr-1 text-orange-400" />
                          {entry.text}
                        </p>
                      </div>
                    )}
                  </div>

                  {/* Stats Summary */}
                  {entry.stats && Object.keys(entry.stats).length > 0 && (
                    <div className="border-t border-gray-600 pt-4">
                      <p className="text-xs text-gray-500 mb-2">Statistics</p>
                      <div className="grid grid-cols-4 gap-3">
                        {Object.entries(entry.stats).map(([key, value]) => {
                          if (key === 'filters' || typeof value !== 'number') return null;
                          return (
                            <div key={key} className="bg-gray-800/50 rounded p-2">
                              <p className="text-xs text-gray-400">{key.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())}</p>
                              <p className="text-sm font-medium text-white">{value}</p>
                            </div>
                          );
                        })}
                        {entry.processingTime && (
                          <div className="bg-gray-800/50 rounded p-2">
                            <p className="text-xs text-gray-400">Processing Time</p>
                            <p className="text-sm font-medium text-white">{entry.processingTime}</p>
                          </div>
                        )}
                      </div>
                    </div>
                  )}

                  {/* First few results preview */}
                  {entry.results && entry.results.length > 0 && (
                    <div className="border-t border-gray-600 pt-4 mt-4">
                      <p className="text-xs text-gray-500 mb-2">Sample Detections</p>
                      <div className="grid grid-cols-5 gap-2">
                        {entry.results.slice(0, 5).map((result, idx) => (
                          <div key={idx} className="relative group">
                            {result.screenshotUrl || result.thumbnail || result.image_path ? (
                              <img 
                                src={result.screenshotUrl || result.thumbnail || 
                                  (result.image_path ? `https://workingcart.com/Police-Surveillance-AI/backend/ai-python/${result.image_path}` : null)
                                }
                                alt="Detection"
                                className="w-full h-16 object-cover rounded-lg border border-gray-600"
                                onError={(e) => {
                                  e.target.onerror = null;
                                  e.target.src = 'https://via.placeholder.com/100?text=No+Image';
                                }}
                              />
                            ) : (
                              <div className="w-full h-16 bg-gray-700 rounded-lg flex items-center justify-center">
                                <Eye className="h-5 w-5 text-gray-500" />
                              </div>
                            )}
                            <div className="absolute inset-0 bg-black/75 opacity-0 group-hover:opacity-100 transition-opacity rounded-lg flex items-center justify-center">
                              <span className="text-xs text-white text-center px-1">
                                {result.confidencePercentage || Math.round(result.confidence * 100) || 0}%
                                <br/>
                                <span className="text-xs text-gray-300">{result.object || 'Object'}</span>
                              </span>
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
    </div>
  );
};

export default PreviousResultsTable;