import React, { useState } from 'react';
import { Shield, LogOut, User, ChevronDown } from 'lucide-react';


const Header = ({ user, onLogout }) => {
  const [showUserMenu, setShowUserMenu] = useState(false);

  const handleLogout = () => {
    console.log("Logout button clicked"); // Debug log
    console.log("onLogout function:", onLogout); // Check if function exists
    
    if (onLogout) {
      console.log("Calling onLogout...");
      onLogout();
    } else {
      console.error("onLogout is undefined!");
    }
    
    setShowUserMenu(false); // Close menu after logout
  };

  return (
    <header className="bg-gray-900 border-b border-gray-700 shadow-lg">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center py-6">
          <div className="flex items-center space-x-3">
            <Shield className="h-8 w-8 text-blue-400" />
            <h1 className="text-2xl font-bold text-white">Police Surveillance AI</h1>
          </div>
          
          <div className="flex items-center space-x-4">
            <span className="px-3 py-1 bg-green-500/10 text-green-400 rounded-full text-sm border border-green-500/20">
              System Online
            </span>
            
            {/* User Menu */}
            <div className="relative">
              <button
                onClick={() => setShowUserMenu(!showUserMenu)}
                className="flex items-center space-x-2 focus:outline-none"
              >
                <div className="h-8 w-8 rounded-full bg-gradient-to-r from-blue-500 to-blue-600 flex items-center justify-center">
                  <User className="h-4 w-4 text-white" />
                </div>
                <span className="text-white text-sm hidden md:block">
                  {user?.name || user?.email || 'User'}
                </span>
                <ChevronDown className="h-4 w-4 text-gray-400" />
              </button>

              {/* Dropdown Menu */}
              {showUserMenu && (
                <div className="absolute right-0 mt-2 w-48 bg-gray-800 rounded-lg shadow-lg border border-gray-700 py-1 z-50">
                  <div className="px-4 py-2 border-b border-gray-700">
                    <p className="text-sm text-white">{user?.name || 'User'}</p>
                    <p className="text-xs text-gray-400">{user?.email || 'No email'}</p>
                  </div>
                  <button
                    onClick={handleLogout}
                    className="w-full text-left px-4 py-2 text-sm text-gray-300 hover:bg-gray-700 flex items-center space-x-2"
                  >
                    <LogOut className="h-4 w-4" />
                    <span>Sign Out</span>
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;