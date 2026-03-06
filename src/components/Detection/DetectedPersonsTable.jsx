// Updated DetectedPersonsTable component
import React from 'react';
import { Play, Download, Eye } from 'lucide-react';

const DetectedPersonsTable = ({ persons }) => {
  // Map of colors to their actual classes
  const colorClasses = {
    blue: {
      bg: 'bg-blue-500/20',
      text: 'text-blue-400',
      border: 'border-blue-500/30'
    },
    yellow: {
      bg: 'bg-yellow-500/20',
      text: 'text-yellow-400',
      border: 'border-yellow-500/30'
    },
    green: {
      bg: 'bg-green-500/20',
      text: 'text-green-400',
      border: 'border-green-500/30'
    },
    purple: {
      bg: 'bg-purple-500/20',
      text: 'text-purple-400',
      border: 'border-purple-500/30'
    }
  };

  const getAttributeBadge = (condition, text, color) => {
    return condition && (
      <span className={`px-2 py-0.5 ${colorClasses[color].bg} ${colorClasses[color].text} rounded text-xs border ${colorClasses[color].border}`}>
        {text}
      </span>
    );
  };

  const handleViewThumbnail = (thumbnailUrl) => {
    if (thumbnailUrl) {
      window.open(thumbnailUrl, '_blank');
    }
  };

  if (!persons || persons.length === 0) {
    return (
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl border border-gray-700 p-8 text-center">
        <p className="text-gray-400">No detection data available</p>
      </div>
    );
  }

  return (
    <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl border border-gray-700">
      <div className="p-4 border-b border-gray-700">
        <h3 className="text-lg font-semibold text-white">Detected Persons</h3>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead className="bg-gray-700/50">
            <tr>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Tracking ID</th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Timestamp</th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Object Type</th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Confidence</th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-700">
            {persons.map((person) => (
              <tr key={person.id} className="hover:bg-gray-700/50 transition-colors">
                <td className="px-4 py-3 text-white font-mono text-sm">#{person.trackingId || person.id}</td>
                <td className="px-4 py-3 text-white">
                  <div>
                    <span className="text-sm">{person.timestamp || person.startTime}s</span>
                    {person.startTime !== person.endTime && (
                      <span className="text-xs text-gray-400 ml-1">
                        (to {person.endTime}s)
                      </span>
                    )}
                  </div>
                </td>
                <td className="px-4 py-3">
                  <div className="flex space-x-2">
                    {getAttributeBadge(true, 'Person with Helmet', 'yellow')}
                    {getAttributeBadge(person.blueShirt, 'Blue Shirt', 'blue')}
                    {getAttributeBadge(person.motorcycle, 'Motorcycle', 'green')}
                  </div>
                </td>
                <td className="px-4 py-3">
                  <span className={`px-2 py-1 rounded text-xs font-medium ${
                    person.confidence > 80 ? 'bg-green-500/20 text-green-400' : 
                    person.confidence > 60 ? 'bg-yellow-500/20 text-yellow-400' : 
                    'bg-red-500/20 text-red-400'
                  }`}>
                    {person.confidence}%
                  </span>
                </td>
                <td className="px-4 py-3">
                  <div className="flex items-center space-x-2">
                    {person.screenshotUrl && (
                      <button 
                        onClick={() => handleViewThumbnail(person.screenshotUrl)}
                        className="p-2 hover:bg-gray-600 rounded-lg transition-colors"
                        title="View Image"
                      >
                        <Eye className="h-4 w-4 text-gray-400" />
                      </button>
                    )}
                    <button className="p-2 hover:bg-gray-600 rounded-lg transition-colors" title="Play">
                      <Play className="h-4 w-4 text-gray-400" />
                    </button>
                    <button className="p-2 hover:bg-gray-600 rounded-lg transition-colors" title="Download">
                      <Download className="h-4 w-4 text-gray-400" />
                    </button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default DetectedPersonsTable;