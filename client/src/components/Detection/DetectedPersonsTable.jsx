import React, { useState } from "react";
import { Play, Download, Eye, Info, X } from "lucide-react";

const DetectedPersonsTable = ({ persons }) => {
  const [selectedPerson, setSelectedPerson] = useState(null);
  const [showModal, setShowModal] = useState(false);

  console.log(
    persons,
    "fgggcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfgggggggg",
  );

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

  const handleViewDetails = (person) => {
    setSelectedPerson(person);
    setShowModal(true);
  };

  const closeModal = () => {
    setShowModal(false);
    setSelectedPerson(null);
  };

  if (!persons || persons.length === 0) {
    return (
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl border border-gray-700 p-8 text-center">
        <p className="text-gray-400">No detection data available</p>
      </div>
    );
  }

  return (
    <>
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl border border-gray-700">
        <div className="p-4 border-b border-gray-700">
          <h3 className="text-lg font-semibold text-white">Detected Objects</h3>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-700/50">
              <tr>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                  Tracking ID
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                  Image
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                  Timestamp (s)
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                  Object Type
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                  Attributes
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                  Confidence
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-700">
              {persons.map((person) => (
                <tr
                  key={`${person.trackingId}_${person.timestamp}`}
                  className="hover:bg-gray-700/50 transition-colors"
                >
                  <td className="px-4 py-3 text-white font-mono text-sm">
                    #{person.trackingId}
                  </td>

                  <td className="px-6 py-4 whitespace-nowrap">
                    {person.image_path ? (
                      <img
                        src={`https://workingcart.com/files/${person.image_path}`}
                        alt={`Person ${person.id}`}
                        className="h-16 w-16 object-cover rounded-lg border border-gray-700 cursor-pointer hover:opacity-80 transition-opacity"
                        onClick={() => handleViewDetails(person)}
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
                      {person.timestamp || person.startTime}s
                    </span>
                  </td>
                  <td className="px-4 py-3">
                    <span className="text-white text-sm">
                      {person.object || "Person"}
                    </span>
                  </td>
                  <td className="px-4 py-3">
                    <div className="flex flex-wrap gap-1">
                      {getAttributeBadge(
                        person.helmet || person.object?.includes("helmet"),
                        "Helmet",
                        "yellow",
                      )}
                      {getAttributeBadge(
                        person.blackShirt || person.object?.includes("black"),
                        "Black",
                        "purple",
                      )}
                      {getAttributeBadge(
                        person.shirt || person.object?.includes("shirt"),
                        "Shirt",
                        "green",
                      )}
                      {getAttributeBadge(
                        person.blueShirt,
                        "Blue Shirt",
                        "blue",
                      )}
                      {getAttributeBadge(
                        person.motorcycle,
                        "Motorcycle",
                        "green",
                      )}
                      {!person.helmet &&
                        !person.blueShirt &&
                        !person.motorcycle &&
                        !person.blackShirt &&
                        !person.shirt && (
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
                          person.confidencePercentage >= 72
                            ? "bg-green-500/20 text-green-400"
                            : person.confidencePercentage > 60
                              ? "bg-yellow-500/20 text-yellow-400"
                              : "bg-red-500/20 text-red-400"
                        }`}
                      >
                        {person.confidencePercentage}%
                      </span>
                    </div>
                  </td>
                  <td className="px-4 py-3">
                    <div className="flex items-center space-x-2">
                      <button
                        onClick={() => handleViewDetails(person)}
                        className="p-2 hover:bg-blue-600/20 rounded-lg transition-colors group"
                        title="View Details"
                      >
                        <Eye className="h-4 w-4 text-blue-400 group-hover:text-blue-300" />
                      </button>
                      {/* {person.bbox && (
                        <button
                          className="p-2 hover:bg-gray-600 rounded-lg transition-colors"
                          title="View Bounding Box"
                        >
                          <Info className="h-4 w-4 text-gray-400" />
                        </button>
                      )} */}
                      {/* <button
                        className="p-2 hover:bg-gray-600 rounded-lg transition-colors"
                        title="Play"
                      >
                        <Play className="h-4 w-4 text-gray-400" />
                      </button>
                      <button
                        className="p-2 hover:bg-gray-600 rounded-lg transition-colors"
                        title="Download"
                      >
                        <Download className="h-4 w-4 text-gray-400" />
                      </button> */}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Details Modal */}
      {showModal && selectedPerson && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm">
          <div className="bg-gray-800 rounded-xl border border-gray-700 max-w-2xl w-full max-h-[90vh] overflow-y-auto">
            <div className="p-4 border-b border-gray-700 flex justify-between items-center sticky top-0 bg-gray-800">
              <h3 className="text-lg font-semibold text-white">
                Detection Details - #{selectedPerson.trackingId}
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
                  {selectedPerson.image_path || selectedPerson.thumbnail ? (
                    <img
                      src={`https://workingcart.com/files/${selectedPerson.image_path}`}
                      alt={`Person ${selectedPerson.id}`}
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
                    #{selectedPerson.trackingId}
                  </p>
                </div>

                <div className="space-y-1">
                  <p className="text-xs text-gray-500">Object ID</p>
                  <p className="text-sm font-medium text-white font-mono">
                    {selectedPerson.id || "N/A"}
                  </p>
                </div>

                <div className="space-y-1">
                  <p className="text-xs text-gray-500">Timestamp</p>
                  <p className="text-sm font-medium text-white">
                    {selectedPerson.timestamp ||
                      selectedPerson.startTime ||
                      "N/A"}
                    s
                  </p>
                </div>

                <div className="space-y-1">
                  <p className="text-xs text-gray-500">Object Type</p>
                  <p className="text-sm font-medium text-white">
                    {selectedPerson.object || "Person"}
                  </p>
                </div>

                <div className="space-y-1">
                  <p className="text-xs text-gray-500">Confidence</p>
                  <span
                    className={`inline-block px-2 py-1 rounded text-xs font-medium ${
                      selectedPerson.confidencePercentage > 80
                        ? "bg-green-500/20 text-green-400"
                        : selectedPerson.confidencePercentage > 60
                          ? "bg-yellow-500/20 text-yellow-400"
                          : "bg-red-500/20 text-red-400"
                    }`}
                  >
                    {selectedPerson.confidencePercentage}%
                  </span>
                </div>

                <div className="space-y-1">
                  <p className="text-xs text-gray-500">Bounding Box</p>
                  <p className="text-sm font-medium text-white">
                    {selectedPerson.bbox
                      ? JSON.stringify(selectedPerson.bbox)
                      : "N/A"}
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
                    selectedPerson.helmet ||
                      selectedPerson.object?.includes("helmet"),
                    "Helmet",
                    "yellow",
                  )}
                  {getAttributeBadge(
                    selectedPerson.blackShirt ||
                      selectedPerson.object?.includes("black"),
                    "Black",
                    "purple",
                  )}
                  {getAttributeBadge(
                    selectedPerson.shirt ||
                      selectedPerson.object?.includes("shirt"),
                    "Shirt",
                    "green",
                  )}
                  {getAttributeBadge(
                    selectedPerson.blueShirt,
                    "Blue Shirt",
                    "blue",
                  )}
                  {getAttributeBadge(
                    selectedPerson.motorcycle,
                    "Motorcycle",
                    "green",
                  )}
                  {!selectedPerson.helmet &&
                    !selectedPerson.blueShirt &&
                    !selectedPerson.motorcycle &&
                    !selectedPerson.blackShirt &&
                    !selectedPerson.shirt && (
                      <span className="text-xs text-gray-500">
                        No attributes
                      </span>
                    )}
                </div>
              </div>

              {/* Additional Data */}
              {/* <div className="space-y-2">
                <h4 className="text-sm font-medium text-gray-400">
                  Additional Information
                </h4>
                <div className="bg-gray-900/50 rounded-lg p-3 space-y-2">
                  {selectedPerson.image_path && (
                    <div>
                      <p className="text-xs text-gray-500">Image Path</p>
                      <p className="text-xs text-gray-400 break-all">
                        {selectedPerson.image_path}
                      </p>
                    </div>
                  )}
                  {selectedPerson.image_path && (
                    <div>
                      <p className="text-xs text-gray-500">Screenshot URL</p>
                      <p className="text-xs text-gray-400 break-all">
                        {`https://workingcart.com/files/${selectedPerson.image_path}`}
                      </p>
                    </div>
                  )}
                  {selectedPerson.object && (
                    <div>
                      <p className="text-xs text-gray-500">Object String</p>
                      <p className="text-xs text-gray-400">
                        {selectedPerson.object}
                      </p>
                    </div>
                  )}
                </div>
              </div> */}
              
              
              {/* Action Buttons */}
              <div className="flex justify-end space-x-3 pt-4 border-t border-gray-700">
                {selectedPerson.image_path && (
                  <button
                    onClick={() =>
                      handleViewThumbnail(
                        `https://workingcart.com/files/${selectedPerson.image_path}`,
                      )
                    }
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
