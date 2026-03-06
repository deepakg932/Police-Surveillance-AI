import React, { useState } from 'react';
import { Search } from 'lucide-react';

const SearchForm = ({ onSearch }) => {
  const [formData, setFormData] = useState({
    timeFrom: '',
    timeTo: '',
    attribute: 'All Persons',
    color: 'All Colors',
    vehicle: 'All Vehicles',
    query: ''
  });

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    onSearch(formData);
  };

  return (
    <div className="bg-gray-800/50 backdrop-blur-sm rounded-2xl border border-gray-700 p-6">
      <h2 className="text-xl font-bold text-white mb-6">Search Database</h2>
      
      <form onSubmit={handleSubmit}>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          <div>
            <label className="block text-sm text-gray-400 mb-2">Time Range</label>
            <div className="flex space-x-2">
              <input 
                type="text" 
                name="timeFrom"
                value={formData.timeFrom}
                onChange={handleChange}
                placeholder="From"
                className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-blue-500"
              />
              <input 
                type="text" 
                name="timeTo"
                value={formData.timeTo}
                onChange={handleChange}
                placeholder="To"
                className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-blue-500"
              />
            </div>
          </div>
          <div>
            <label className="block text-sm text-gray-400 mb-2">Attributes</label>
            <div className="flex space-x-2">
              <select 
                name="attribute"
                value={formData.attribute}
                onChange={handleChange}
                className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-blue-500"
              >
                <option>All Persons</option>
                <option>With Helmet</option>
                <option>Without Helmet</option>
              </select>
              <select 
                name="color"
                value={formData.color}
                onChange={handleChange}
                className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-blue-500"
              >
                <option>All Colors</option>
                <option>Blue Shirt</option>
                <option>Red Shirt</option>
                <option>Black Shirt</option>
              </select>
            </div>
          </div>
          <div>
            <label className="block text-sm text-gray-400 mb-2">Vehicle</label>
            <select 
              name="vehicle"
              value={formData.vehicle}
              onChange={handleChange}
              className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-blue-500"
            >
              <option>All Vehicles</option>
              <option>With Motorcycle</option>
              <option>Without Motorcycle</option>
            </select>
          </div>
        </div>

        <div className="relative mb-6">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-500" />
          <input
            type="text"
            name="query"
            value={formData.query}
            onChange={handleChange}
            placeholder="Enter search query... (e.g., 'Person with blue shirt on motorcycle')"
            className="w-full pl-10 pr-4 py-3 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-blue-500"
          />
        </div>

        <button 
          type="submit"
          className="w-full py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium transition-colors flex items-center justify-center space-x-2"
        >
          <Search className="h-5 w-5" />
          <span>Execute Search Query</span>
        </button>
      </form>
    </div>
  );
};

export default SearchForm;