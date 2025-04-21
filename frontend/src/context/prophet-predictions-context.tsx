import { createContext, type ReactNode, useContext, useEffect, useState } from 'react';

import { fetchProphetPredictions } from '@/api/backendApi';
import { type ProphetPredictionData } from '@/types/types';

interface ProphetPredictionsContextType {
  data: ProphetPredictionData[] | null;
  isLoading: boolean;
  error: Error | null;
  fetchPredictions: () => Promise<void>;
  lastUpdated: number | null;
}

const ProphetPredictionsContext = createContext<ProphetPredictionsContextType | undefined>(
  undefined
);

export function ProphetPredictionsProvider({ children }: { children: ReactNode }) {
  const [data, setData] = useState<ProphetPredictionData[] | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [lastUpdated, setLastUpdated] = useState<number | null>(null);

  const fetchPredictions = async () => {
    setIsLoading(true);
    setError(null);

    try {
      const result = await fetchProphetPredictions();

      if (result instanceof Error) {
        setError(result);
      } else {
        if (!data || !arraysEqual(data, result)) {
          setData(result);
          setLastUpdated(Date.now());
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Failed to fetch predictions'));
    } finally {
      setIsLoading(false);
    }
  };

  // Helper function to compare arrays
  const arraysEqual = (arr1: ProphetPredictionData[], arr2: ProphetPredictionData[]): boolean => {
    if (arr1.length !== arr2.length) return false;

    // Simple check based on JSON representation
    return JSON.stringify(arr1) === JSON.stringify(arr2);
  };

  // Initial fetch on mount
  useEffect(() => {
    void fetchPredictions();
  }, []);

  return (
    <ProphetPredictionsContext.Provider
      value={{ data, isLoading, error, fetchPredictions, lastUpdated }}
    >
      {children}
    </ProphetPredictionsContext.Provider>
  );
}

export function useProphetPredictionsContext() {
  const context = useContext(ProphetPredictionsContext);
  if (context === undefined) {
    throw new Error(
      'useProphetPredictionsContext must be used within a ProphetPredictionsProvider'
    );
  }
  return context;
}
