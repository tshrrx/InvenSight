export default function MainPage() {
  return (
    <div className="min-h-screen bg-black text-white">
      <div className="container mx-auto px-4 py-8">
        <h1 className="text-4xl font-bold mb-8">Query Your Data</h1>
        <div className="space-y-4">
          <input
            type="text"
            placeholder="Ask a question about your retail data..."
            className="w-full px-4 py-3 rounded-lg bg-gray-900 text-white border border-gray-700 focus:outline-none focus:border-white"
          />
          <button className="px-6 py-3 bg-white text-black font-semibold rounded-lg hover:bg-gray-200 transition-colors duration-300">
            Submit Query
          </button>
        </div>
        <div className="mt-8 p-6 bg-gray-900 rounded-lg">
          <h2 className="text-2xl font-semibold mb-4">Results</h2>
          <p className="text-gray-400">Your query results will appear here...</p>
        </div>
      </div>
    </div>
  );
}