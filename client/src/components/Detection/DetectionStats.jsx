import React from 'react';
import { 
  User, Shield, Clock, Users, 
  Shirt, Circle, Eye, AlertCircle,
  Car, Bike, Filter, Activity
} from 'lucide-react';

const DetectionStats = ({ stats, mode }) => {
  // Default stats if not provided
  const {
    totalPersons = 0,
    totalHelmet = 0,
    totalVehicles = 0,
    totalDetections = 0,
    processingTime = 'N/A',
    filters = {}
  } = stats || {};

  // Format processing time
  const formatProcessingTime = (time) => {
    if (time === 'N/A' || !time) return 'N/A';
    if (typeof time === 'string' && time.includes('s')) return time;
    return `${time}s`;
  };

  // Check if there are any active filters
  const hasActiveFilters = Object.values(filters).some(value => 
    value !== null && value !== undefined && value !== ''
  );

  // Map count keys to icons and colors
  const getIconForStat = (key) => {
    const iconMap = {
      'totalPersons': Users,
      'totalHelmet': Shield,
      'totalVehicles': Car,
      'totalDetections': Activity,
      'withHelmet': Shield,
      'total_person': Users,
      'total_helmet': Shield,
      'total_vehicle': Car,
      'total_car': Car,
      'total_bike': Bike,
      'total_wear': Shield,
      'total_black': Circle,
      'total_shirt': Shirt
    };
    return iconMap[key] || Eye;
  };

  const getColorForStat = (key) => {
    const colorMap = {
      'totalPersons': 'blue',
      'totalHelmet': 'yellow',
      'totalVehicles': 'green',
      'totalDetections': 'purple',
      'withHelmet': 'yellow',
      'total_person': 'blue',
      'total_helmet': 'yellow',
      'total_vehicle': 'green',
      'total_car': 'green',
      'total_bike': 'green',
      'total_wear': 'yellow',
      'total_black': 'purple',
      'total_shirt': 'green'
    };
    return colorMap[key] || 'gray';
  };

  const getLabelForStat = (key) => {
    const labelMap = {
      'totalPersons': 'Total Persons',
      'totalHelmet': 'With Helmet',
      'totalVehicles': 'Total Vehicles',
      'totalDetections': 'Total Detections',
      'withHelmet': 'With Helmet',
      'total_person': 'Total Persons',
      'total_helmet': 'With Helmet',
      'total_vehicle': 'Total Vehicles',
      'total_car': 'Total Cars',
      'total_bike': 'Total Bikes',
      'total_wear': 'With Wear',
      'total_black': 'With Black',
      'total_shirt': 'With Shirt'
    };
    return labelMap[key] || key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
  };

  // Map of colors to their actual classes
  const colorClasses = {
    blue: {
      bg: 'bg-blue-500/10',
      text: 'text-blue-400',
      border: 'border-blue-500/20'
    },
    green: {
      bg: 'bg-green-500/10',
      text: 'text-green-400',
      border: 'border-green-500/20'
    },
    yellow: {
      bg: 'bg-yellow-500/10',
      text: 'text-yellow-400',
      border: 'border-yellow-500/20'
    },
    purple: {
      bg: 'bg-purple-500/10',
      text: 'text-purple-400',
      border: 'border-purple-500/20'
    },
    gray: {
      bg: 'bg-gray-500/10',
      text: 'text-gray-400',
      border: 'border-gray-500/20'
    }
  };

  // Get all stats that start with 'total_' or are standard stats
  const getMainStats = () => {
    const mainStatKeys = ['totalDetections', 'totalPersons', 'totalHelmet', 'totalVehicles'];
    return mainStatKeys
      .filter(key => stats[key] !== undefined)
      .map(key => ({
        label: getLabelForStat(key),
        value: stats[key],
        icon: getIconForStat(key),
        color: getColorForStat(key),
        key
      }));
  };

  // Get additional stats (any other total_* fields)
  const getAdditionalStats = () => {
    return Object.entries(stats)
      .filter(([key, value]) => {
        // Filter for total_* fields that aren't in main stats and have positive values
        return key.startsWith('total_') && 
               !['totalPersons', 'totalHelmet', 'totalVehicles', 'totalDetections'].includes(key) &&
               typeof value === 'number' &&
               value > 0;
      })
      .map(([key, value]) => ({
        label: getLabelForStat(key),
        value,
        icon: getIconForStat(key),
        color: getColorForStat(key),
        key
      }));
  };

  const mainStats = getMainStats();
  const additionalStats = getAdditionalStats();

  return (
    <div className="space-y-6">
      {/* Mode Badge */}
      {mode && (
        <div className="flex items-center justify-between">
          <span className="px-3 py-1 bg-blue-500/20 text-blue-400 rounded-full text-sm font-medium">
            Mode: {mode}
          </span>
          {hasActiveFilters && (
            <div className="flex items-center space-x-2">
              <Filter className="h-4 w-4 text-gray-400" />
              <span className="text-sm text-gray-400">Filters Applied</span>
            </div>
          )}
        </div>
      )}

      {/* Main Stats Grid */}
      {mainStats.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {mainStats.map(({ label, value, icon: Icon, color }) => (
            <div 
              key={label} 
              className={`bg-gray-800/50 backdrop-blur-sm rounded-xl border ${colorClasses[color]?.border || 'border-gray-700'} p-4 hover:bg-gray-800/70 transition-all`}
            >
              <div className="flex items-center space-x-3">
                <div className={`p-2 ${colorClasses[color]?.bg || colorClasses.gray.bg} rounded-lg`}>
                  <Icon className={`h-5 w-5 ${colorClasses[color]?.text || colorClasses.gray.text}`} />
                </div>
                <div>
                  <p className="text-sm text-gray-400">{label}</p>
                  <p className="text-2xl font-bold text-white">{value}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Processing Time & Filters Row */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Processing Time Card */}
        {processingTime && (
          <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl border border-gray-700 p-4">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-purple-500/10 rounded-lg">
                <Clock className="h-5 w-5 text-purple-400" />
              </div>
              <div>
                <p className="text-sm text-gray-400">Processing Time</p>
                <p className="text-xl font-bold text-white">{formatProcessingTime(processingTime)}</p>
              </div>
            </div>
          </div>
        )}

        {/* Filters Card */}
        {hasActiveFilters && (
          <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl border border-gray-700 p-4">
            <div className="flex items-start space-x-3">
              <div className="p-2 bg-blue-500/10 rounded-lg">
                <Filter className="h-5 w-5 text-blue-400" />
              </div>
              <div className="flex-1">
                <p className="text-sm text-gray-400 mb-2">Active Filters</p>
                <div className="flex flex-wrap gap-2">
                  {Object.entries(filters).map(([key, value]) => {
                    if (value === null || value === undefined || value === '') return null;
                    return (
                      <span 
                        key={key}
                        className="px-2 py-1 bg-blue-500/20 text-blue-400 rounded-md text-xs font-medium"
                      >
                        {key}: {value.toString()}
                      </span>
                    );
                  })}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Additional Stats */}
      {additionalStats.length > 0 && (
        <div className="mt-6">
          <h3 className="text-sm font-medium text-gray-400 mb-3">Additional Statistics</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {additionalStats.map(({ label, value, icon: Icon, color }) => (
              <div 
                key={label}
                className="bg-gray-800/30 backdrop-blur-sm rounded-lg border border-gray-700 p-3"
              >
                <div className="flex items-center space-x-2">
                  <div className={`p-1.5 ${colorClasses[color]?.bg || colorClasses.gray.bg} rounded`}>
                    <Icon className={`h-4 w-4 ${colorClasses[color]?.text || colorClasses.gray.text}`} />
                  </div>
                  <div>
                    <p className="text-xs text-gray-400">{label}</p>
                    <p className="text-lg font-semibold text-white">{value}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* No Stats State */}
      {mainStats.length === 0 && additionalStats.length === 0 && !processingTime && (
        <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl border border-gray-700 p-8 text-center">
          <AlertCircle className="h-12 w-12 text-gray-600 mx-auto mb-3" />
          <p className="text-gray-400">No statistics available</p>
        </div>
      )}
    </div>
  );
};

export default DetectionStats;