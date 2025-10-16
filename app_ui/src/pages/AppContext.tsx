import { createContext, useContext, useState } from "react";

interface AppContextType {
  error: string | null;
  setError: (error: string | null) => void;
  columns: string[];
  setColumns: (column: string[]) => void;
  query: string;
  setQuery: (query: string) => void;
  result: string;
  setResult: (result: string) => void;
  spinner: boolean;
  setSpinner: (spinner: boolean) => void;
}

const AppContext = createContext<AppContextType | undefined>(undefined);

export const AppContextProvider: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  const [error, setError] = useState<string | null>(null);
  const [columns, setColumns] = useState<string[]>([]);
  const [query, setQuery] = useState("");
  const [result, setResult] = useState("");
  const [spinner, setSpinner] = useState<boolean>(false);

  return (
    <AppContext.Provider
      value={{
        query,
        setQuery,
        columns,
        setColumns,
        result,
        setResult,
        spinner,
        setSpinner,
        error,
        setError,
      }}
    >
      {children}
    </AppContext.Provider>
  );
};

export const useAppContext = (): AppContextType => {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error("useAppContext must be used within AppProvider");
  }
  return context;
};
