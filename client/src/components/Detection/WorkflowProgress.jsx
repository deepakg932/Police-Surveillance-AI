import React from "react";
import { useRef, useEffect } from "react";
import { Loader2, Radio, Search, Download } from "lucide-react";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL;

const WorkflowProgress = ({
  job = {},
  results = [],
  status = "idle",
  apiBaseUrl = API_BASE_URL,
}) => {
  const safeResults = Array.isArray(results) ? results : [];

  const latestResult =
    safeResults.length > 0
      ? [...safeResults].sort(
          (a, b) => (b.confidence || 0) - (a.confidence || 0),
        )[0]
      : null;

  // const latestResult = safeResults[safeResults.length - 1] || null;

  const isProcessing = status === "processing";
  const isCompleted = status === "completed" && latestResult;

  const prompt = job?.textNote || job?.prompt || "No query provided";
  const fileName = job?.fileName || job?.videoName || "CCTV_Footage.mp4";

  const imageUrl =
    latestResult?.image_path ||
    latestResult?.screenshotUrl?.split("/").pop() ||
    "";
  const matchImageUrl = imageUrl
    ? `https://workingcart.com/files/${imageUrl}`
    : "";
  const confidence = Math.round((latestResult?.confidence || 0) * 100);

  const keywords = prompt
    .toLowerCase()
    .split(" ")
    .filter((w) =>
      [
        "person",
        "bike",
        "helmet",
        "car",
        "bus",
        "truck",
        "scooter",
        "bag",
      ].includes(w),
    );

  const videoUrlRef = useRef(null);

  useEffect(() => {
    if (job.currentVideoFile) {
      // purge old
      if (videoUrlRef.current) {
        URL.revokeObjectURL(videoUrlRef.current);
      }

      // create new
      videoUrlRef.current = URL.createObjectURL(job.currentVideoFile);
    }

    return () => {
      if (videoUrlRef.current) {
        URL.revokeObjectURL(videoUrlRef.current);
      }
    };
  }, [job.currentVideoFile]);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-5 bg-gray-900/70 border border-blue-500/30 rounded-xl p-5">
      {/* Panel 1 */}
      <div className="bg-gray-800/70 border border-gray-700 rounded-xl overflow-hidden">
        <div className="p-3 border-b border-gray-700 flex items-center gap-2 text-blue-300 text-xs font-semibold">
          <span className="w-5 h-5 rounded-full bg-blue-500 flex items-center justify-center text-white">
            1
          </span>
          INPUT SOURCE
          {isProcessing && (
            <span className="ml-auto text-red-400 animate-pulse">● REC</span>
          )}
        </div>

        <div className="h-48 bg-black relative flex items-center justify-center overflow-hidden">
          <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.04)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.04)_1px,transparent_1px)] bg-[size:20px_20px]" />

          {isProcessing && (
            <div className="absolute top-0 left-0 w-full h-1 bg-blue-400 animate-pulse" />
          )}

          {/* ✅ YEH PART ADD KARNA HAI */}
          {job.currentVideoFile || job.videoPreviewUrl ? (
            <video
              key={job?.id || "video"} // important
              src={videoUrlRef.current || job.videoPreviewUrl}
              className="relative z-10 w-full h-full object-cover"
              controls
              muted
              playsInline
            />
          ) : (
            <div className="relative z-10 text-gray-400 text-sm text-center">
              CCTV INPUT READY
            </div>
          )}
        </div>

        <div className="p-3 text-xs text-gray-400 truncate">
          {fileName}
          {job.progressMsg && (
            <div className="text-blue-300 mt-1">{job.progressMsg}</div>
          )}
        </div>
      </div>

      {/* Panel 2 */}
      <div className="bg-gray-800/70 border border-gray-700 rounded-xl overflow-hidden">
        <div className="p-3 border-b border-gray-700 flex items-center gap-2 text-blue-300 text-xs font-semibold">
          <span className="w-5 h-5 rounded-full bg-blue-500 flex items-center justify-center text-white">
            2
          </span>
          ANALYSIS QUERY
        </div>

        <div className="p-4 space-y-4">
          <div className="border border-blue-500/40 bg-blue-500/10 rounded-lg p-3 text-center text-white text-sm">
            {job.queryImagePreviewUrl ? (
              <div className="flex flex-col items-center gap-2">
                <div
                  className={`flex items-center gap-2 ${
                    isCompleted
                      ? "text-green-400"
                      : isProcessing
                        ? "text-blue-300"
                        : "text-gray-300"
                  }`}
                >
                  {isCompleted ? (
                    <>✓ Image Matched</>
                  ) : isProcessing ? (
                    <>
                      <Loader2 className="h-4 w-4 animate-spin" />
                      Scanning Image
                    </>
                  ) : (
                    <>Reference Image</>
                  )}
                </div>

                <img
                  src={job.queryImagePreviewUrl}
                  alt="Query Reference"
                  className="max-h-32 w-full object-contain rounded-lg border border-blue-500/30"
                />
              </div>
            ) : isProcessing ? (
              <span className="flex items-center justify-center gap-2 text-blue-300">
                <Loader2 className="h-4 w-4 animate-spin" />
                Scanning: {prompt}
              </span>
            ) : (
              <span className="italic">{prompt}</span>
            )}
          </div>

          <div className="flex gap-2">
            <span className="px-2 py-1 rounded bg-blue-500/20 text-blue-300 text-xs">
              ✓ Syntax Valid
            </span>
            <span className="px-2 py-1 rounded bg-blue-500/20 text-blue-300 text-xs">
              {keywords.length || 0} Objects
            </span>
          </div>

          <div className="text-xs text-gray-400 space-y-1">
            <p className="text-gray-500">AI ENGINE PARSING:</p>
            {keywords.length > 0 ? (
              keywords.map((key) => <p key={key}>› Attribute: {key}</p>)
            ) : (
              <p>› Waiting for query keywords...</p>
            )}
          </div>
        </div>
      </div>

      {/* Panel 3 */}
      <div className="bg-gray-800/70 border border-blue-500/50 rounded-xl overflow-hidden shadow-lg shadow-blue-500/10">
        <div className="p-3 border-b border-gray-700 flex items-center gap-2 text-blue-300 text-xs font-semibold">
          <span className="w-5 h-5 rounded-full bg-blue-500 flex items-center justify-center text-white">
            3
          </span>
          MATCH DETECTED
          {isCompleted && (
            <span className="ml-auto px-2 py-0.5 bg-green-500 text-black rounded text-[10px]">
              FOUND
            </span>
          )}
        </div>

        <div className="h-48 bg-black relative flex items-center justify-center">
          {imageUrl ? (
            <img
              src={`https://workingcart.com/files/${imageUrl}`}
              alt="Live Match"
              className="max-h-full max-w-full object-contain border border-green-500"
            />
          ) : isProcessing ? (
            <div className="flex flex-col items-center text-blue-300">
              <Loader2 className="h-12 w-12 animate-spin mb-3" />
              <p className="text-xs">Scanning frames...</p>
            </div>
          ) : (
            <div className="text-gray-500 text-sm flex flex-col items-center">
              <Search className="h-10 w-10 mb-2" />
              Waiting for match
            </div>
          )}

          {/* {!isProcessing && !isCompleted && (
            <div className="text-gray-500 text-sm flex flex-col items-center">
              <Search className="h-10 w-10 mb-2" />
              Waiting for match
            </div>
          )} */}
        </div>

        <div className="grid grid-cols-2 gap-3 p-3 text-xs">
          <div>
            <p className="text-gray-500">TIMESTAMP</p>
            <p className="text-white">{latestResult?.timestamp || "00:00"}</p>
          </div>
          <div>
            <p className="text-gray-500">CONFIDENCE</p>
            <p className="text-green-400">{confidence}%</p>
          </div>
          <div>
            <p className="text-gray-500">TRACKING ID</p>
            <p className="text-white">
              {latestResult?.trackingId
                ? `#TRK-${latestResult.trackingId.slice(-4)}`
                : "N/A"}
            </p>
          </div>
          <div>
            <p className="text-gray-500">ACTION</p>
            <a
              href={`${apiBaseUrl}/video/download-image?file=${encodeURIComponent(imageUrl)}`}
              className="text-blue-400 flex items-center gap-1 hover:text-blue-200"
            >
              <Download className="h-3 w-3" /> Download
            </a>
          </div>
        </div>
      </div>
    </div>
  );
};

export default WorkflowProgress;
