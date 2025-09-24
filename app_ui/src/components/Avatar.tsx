import React from "react";

interface AvatarProps {
  role: "user" | "assistant";
}

const Avatar: React.FC<AvatarProps> = ({ role }) => {
  if (role === "user") {
    return (
      <div className="w-9 h-9 rounded-full bg-blue-600 flex items-center justify-center text-white text-2xl shadow">
        <span role="img" aria-label="User">👤</span>
      </div>
    );
  }
  return (
    <div className="w-9 h-9 rounded-full bg-slate-400 flex items-center justify-center text-white text-2xl shadow">
      <span role="img" aria-label="Assistant">🤖</span>
    </div>
  );
};

export default Avatar;
