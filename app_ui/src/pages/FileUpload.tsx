import { useState } from "react";
import { useAppContext } from "./AppContext";
import api from "../Api";
import { useNavigate } from "react-router-dom";


function FileUpload() {
  const [files, setFiles] = useState<(File | null)[]>([null]);
  const { error, spinner, setSpinner } = useAppContext();
  const [csvFiles, setCsvFiles] = useState<{
    [filename: string]: {
      columns: string[];
      content: string[];
    };
  }>({});
  const navigate = useNavigate();

  const UploadFile = async () => {
    if (files.length === 0) return;
    setSpinner(true);
    try {
      const formData = new FormData();
      files.filter((file): file is File => !!file).forEach((file) => formData.append("files", file));
      const response = await api.post("/upload-file", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      if (response.data.csv_files) {
        // Initialize content selection as all columns for each CSV
        const csvMap: any = {};
        response.data.csv_files.forEach((csv: { filename: string; columns: string[] }) => {
          csvMap[csv.filename] = { columns: csv.columns, content: [...csv.columns] };
        });
        setCsvFiles(csvMap);
      } else {
        setCsvFiles({});
        // If PDF, immediately create vector DB
        await CreateVectorDatabase();
      }
    } catch (err) {
    } finally {
      setSpinner(false);
    }
  };

  const toggleCsvContentColumn = (filename: string, column: string) => {
    setCsvFiles(prev => {
      const file = prev[filename];
      if (!file) return prev;
      const content = file.content.includes(column)
        ? file.content.filter(c => c !== column)
        : [...file.content, column];
      return { ...prev, [filename]: { ...file, content } };
    });
  };

  const CreateVectorDatabase = async () => {
    setSpinner(true);
    try {
      if (Object.keys(csvFiles).length > 0) {
        // Build mapping: filename -> { content, metadata }
        const csv_column_map: any = {};
        Object.entries(csvFiles).forEach(([filename, { columns, content }]) => {
          csv_column_map[filename] = {
            content,
            metadata: columns.filter(col => !content.includes(col)),
          };
        });
        await api.post(
          "/set-columns",
          new URLSearchParams({
            csv_column_map: JSON.stringify(csv_column_map),
          })
        );
      }
      await api.post("/create-vector-database");
      setCsvFiles({});
      setFiles([null]);
  navigate("/");
    } catch (err) {
    } finally {
      setSpinner(false);
    }
  };


  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>, idx: number) => {
    if (e.target.files && e.target.files.length > 0) {
      const newFiles = [...files];
      newFiles[idx] = e.target.files[0];
      setFiles(newFiles);
    }
  };

  return (
    <div className="min-h-screen w-full bg-gradient-to-br from-black via-gray-900 to-red-950 flex flex-col items-center justify-center">

      <div className="bg-black/80 border border-red-900 rounded-2xl shadow-2xl p-10 w-full max-w-xl flex flex-col gap-6 animate-fade-in">
        <label className="block text-white text-lg font-semibold mb-2">Upload PDF or CSV</label>
        {files.map((file, idx) => (
          <div key={idx} className="relative block w-full mb-2 flex items-center gap-2">
            <label className="flex-1">
              <input
                type="file"
                accept=".pdf,.csv"
                className="hidden"
                onChange={e => handleImageChange(e, idx)}
                disabled={spinner}
              />
              <div className="flex items-center">
                <button
                  type="button"
                  className={`w-full bg-black/60 border border-red-900 rounded-lg px-3 py-2 text-white text-left focus:outline-none focus:border-red-600 focus:ring-2 focus:ring-red-900 transition ${spinner ? 'opacity-60 cursor-not-allowed' : 'cursor-pointer hover:bg-red-950'}`}
                  onClick={() => {
                    if (spinner) return;
                    const inputs = document.querySelectorAll('input[type=file]');
                    (inputs[idx] as HTMLInputElement)?.click();
                  }}
                  disabled={spinner}
                >
                  {file ? file.name : "Click to select a file..."}
                </button>
              </div>
            </label>
            {files.length > 1 && (
              <button
                type="button"
                className={`ml-2 px-2 py-1 text-red-400 hover:text-red-600 text-xl rounded focus:outline-none focus:ring-2 focus:ring-red-900 ${spinner ? 'cursor-not-allowed' : 'cursor-pointer'}`}
                onClick={() => setFiles(files.filter((_, i) => i !== idx))}
                disabled={spinner}
                aria-label="Remove file"
                title="Remove file"
              >
                🗑️
              </button>
            )}
          </div>
        ))}
        {files.every(f => f) && (
          <button
            type="button"
            className={`bg-gradient-to-r from-gray-700 to-gray-500 px-4 py-1 text-white rounded-lg shadow hover:from-gray-600 hover:to-gray-400 font-semibold transition mb-2 ${spinner ? 'cursor-not-allowed' : 'cursor-pointer'}`}
            onClick={() => setFiles([...files, null])}
            disabled={spinner}
          >
            + Add another file
          </button>
        )}
        <button
          className={`bg-gradient-to-r from-red-700 to-red-500 px-5 py-2 text-lg text-white rounded-lg shadow hover:from-red-600 hover:to-red-400 font-bold transition ${spinner ? 'cursor-not-allowed' : 'cursor-pointer'}`}
          onClick={UploadFile}
          disabled={spinner || files.length === 0 || files.some(f => !f)}
          style={spinner ? { opacity: 0.6, cursor: 'not-allowed' } : {}}
        >
          {spinner ? "Uploading..." : "Upload"}
        </button>
        {Object.keys(csvFiles).length > 0 && (
          <div className="mt-4">
            <label className="block text-white text-lg font-semibold mb-2">Select columns for each CSV (content columns will be embedded, others as metadata):</label>
            {Object.entries(csvFiles).map(([filename, { columns, content }]) => (
              <div key={filename} className="mb-6 p-4 rounded-lg border border-red-700 bg-black/60">
                <div className="text-red-300 font-bold mb-2">{filename}</div>
                <div className="flex flex-wrap gap-2">
                  {columns.map((col) => (
                    <button
                      key={col}
                      className={`px-3 py-1 rounded-full border-2 text-white font-semibold transition ${content.includes(col) ? "bg-red-700 border-red-700" : "bg-black/60 border-red-900 hover:bg-red-900"}`}
                      onClick={() => toggleCsvContentColumn(filename, col)}
                    >
                      {col}
                    </button>
                  ))}
                </div>
                <div className="text-xs text-gray-400 mt-2">Content columns: <span className="font-mono">{content.join(", ")}</span></div>
                <div className="text-xs text-gray-400">Metadata columns: <span className="font-mono">{columns.filter(col => !content.includes(col)).join(", ") || "(none)"}</span></div>
              </div>
            ))}
            <button
              className={`mt-4 bg-gradient-to-r from-red-700 to-red-500 px-5 py-2 text-lg text-white rounded-lg shadow hover:from-red-600 hover:to-red-400 font-bold transition ${spinner ? 'cursor-not-allowed' : 'cursor-pointer'}`}
              onClick={CreateVectorDatabase}
              disabled={spinner}
              style={spinner ? { opacity: 0.6, cursor: 'not-allowed' } : {}}
            >
              {spinner ? "Processing..." : "Create Vector Database"}
            </button>
          </div>
        )}
        {error && <p className="text-red-400 text-center mt-4 font-semibold">{error}</p>}
        {spinner && (
          <div className="flex justify-center items-center my-4">
            <svg
              className="animate-spin h-6 w-6 text-white"
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 24 24"
            >
              <circle
                className="opacity-25"
                cx="12"
                cy="12"
                r="10"
                stroke="currentColor"
                strokeWidth="4"
              ></circle>
              <path
                className="opacity-75"
                fill="currentColor"
                d="M4 12a8 8 0 018-8v8H4z"
              ></path>
            </svg>
            <p className="ml-2 text-white">Processing...</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default FileUpload;
