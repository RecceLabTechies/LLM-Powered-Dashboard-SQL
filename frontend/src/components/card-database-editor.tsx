'use client';

import { useCallback, useEffect } from 'react';

import { Trash2 } from 'lucide-react';
import { toast } from 'sonner';

import { Button } from '@/components/ui/button';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';

import { useDatabases, useDeleteDatabase } from '@/hooks/use-backend-api';

interface DatabaseEditorCardProps {
  onEditSuccess?: () => void;
}

export default function DatabaseEditorCard({ onEditSuccess }: DatabaseEditorCardProps) {
  const { data: databases, fetchDatabases, isLoading: isFetching } = useDatabases();
  const { deleteDatabase, isLoading: isDeleting } = useDeleteDatabase();
  useEffect(() => {
    void fetchDatabases();
  }, [fetchDatabases]);
  const handleDelete = useCallback(
    async (dbName: string) => {
      try {
        await deleteDatabase(dbName);
        toast.success(`Database "${dbName}" cleared successfully`);

        void fetchDatabases();
        onEditSuccess?.();
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
        toast.error(`Failed to clear database: ${errorMessage}`);
      }
    },
    [deleteDatabase, fetchDatabases, onEditSuccess]
  );

  return (
    <>
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-medium">Clean Databases</h3>
      </div>

      {/* Database Selection and Actions */}

      <div className="space-y-2">
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button
              variant="outline"
              className="w-full justify-between"
              disabled={isFetching || !databases?.length}
              aria-label={
                isFetching
                  ? 'Loading databases...'
                  : `Select a database to clean (${databases?.length ?? 0} available)`
              }
            >
              Select a database
              {isFetching ? (
                <span className="animate-pulse">Loading...</span>
              ) : (
                <span>{databases?.length ?? 0} available</span>
              )}
            </Button>
          </DropdownMenuTrigger>

          <DropdownMenuContent className="w-64">
            {databases?.map((db) => (
              <DropdownMenuItem
                key={db}
                className="group cursor-pointer justify-between"
                onClick={(e) => {
                  e.stopPropagation();
                  void handleDelete(db);
                }}
                aria-label={`Delete database ${db}`}
              >
                {db}
                <Trash2 size={18} aria-hidden="true" />
              </DropdownMenuItem>
            ))}
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
    </>
  );
}
