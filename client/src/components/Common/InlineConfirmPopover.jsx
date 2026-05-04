import React from "react";

const InlineConfirmPopover = ({ message, onConfirm, onCancel, confirmLabel = "Yes", cancelLabel = "No" }) => {
  return (
    <div className="absolute right-0 top-full z-50 mt-0 w-60 rounded-lg border border-gray-600 bg-gray-900 p-3 shadow-xl">
      <p className="text-xs text-gray-200 leading-4">{message}</p>
      <div className="mt-3 flex items-center justify-end gap-2">
        <button
          onClick={onCancel}
          className="rounded-md border border-gray-600 px-2 py-1 text-xs text-gray-300 hover:bg-gray-800"
        >
          {cancelLabel}
        </button>
        <button
          onClick={onConfirm}
          className="rounded-md bg-red-600 px-2 py-1 text-xs text-white hover:bg-red-500"
        >
          {confirmLabel}
        </button>
      </div>
    </div>
  );
};

export default InlineConfirmPopover;
