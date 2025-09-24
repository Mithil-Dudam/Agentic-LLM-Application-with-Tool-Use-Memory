import React from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface MarkdownMessageProps {
  content: string;
}

const MarkdownMessage: React.FC<MarkdownMessageProps> = ({ content }) => {
  return (
    <div className="prose prose-invert max-w-none">
      <ReactMarkdown
        children={content}
        remarkPlugins={[remarkGfm]}
        components={{
          code({ inline, children }: { inline?: boolean; children?: React.ReactNode }) {
            return !inline ? (
              <pre className="bg-gray-900 text-green-200 rounded p-3 overflow-x-auto my-2 text-sm">
                <code>{children}</code>
              </pre>
            ) : (
              <code className="bg-gray-200 text-gray-800 rounded px-1 py-0.5 text-sm">{children}</code>
            );
          },
          li({ children }: { children?: React.ReactNode }) {
            return <li className="ml-4 list-disc">{children}</li>;
          },
        }}
      />
    </div>
  );
};

export default MarkdownMessage;
