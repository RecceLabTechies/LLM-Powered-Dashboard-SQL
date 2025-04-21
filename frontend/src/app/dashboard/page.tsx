'use client';

import { DatabaseOperationsProvider } from '@/context/database-operations-context';
import { ProphetPredictionsProvider } from '@/context/prophet-predictions-context';
import { BarChart3, EyeOff, Globe, Users2 } from 'lucide-react';

import DatabaseUploaderEditorCard from '@/components/card-database-uploader-editor';
import { MetricsPredictedRevenueCard } from '@/components/card-metrics-predicted-revenue';
import { MetricsPredictedROICard } from '@/components/card-metrics-predicted-roi';
import { MetricsRevenueCard } from '@/components/card-metrics-revenue';
import { MetricsROICard } from '@/components/card-metrics-roi';
import { MLTriggerCard } from '@/components/card-ml-trigger';
import ChannelContributionChart from '@/components/chart-channel-contribution';
import { CostMetricsHeatmap } from '@/components/chart-cost-metrics-heatmap';
import { RevenueAdSpendChart } from '@/components/chart-filter-revenue-adspend';
import { MLPredictionsChart } from '@/components/chart-ml-predictions';
import AgeGroupPerformanceCharts from '@/components/charts-adspend-revenue-agegroup';
import ChannelPerformanceCharts from '@/components/charts-adspend-revenue-channels';
import CountryPerformanceCharts from '@/components/charts-adspend-revenue-countries';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';

export default function DashboardPage() {
  return (
    <DatabaseOperationsProvider>
      <ProphetPredictionsProvider>
        <div className="flex flex-col container gap-4 mx-auto pb-4">
          {/* Summary Cards */}
          <section className="grid grid-cols-4 gap-4" aria-label="Key metrics and actions">
            <MetricsRevenueCard />
            <MetricsROICard />
            <MetricsPredictedRevenueCard />
            <DatabaseUploaderEditorCard />
            <MLTriggerCard />
            <MetricsPredictedROICard />
          </section>

          <section className="grid grid-cols-2 gap-4"></section>

          {/* ML Predictions Chart */}
          <section aria-label="Machine learning predictions">
            <MLPredictionsChart />
          </section>

          {/* Every Revenue & Ad Spend Chart */}
          <section aria-label="Performance charts by category">
            <Tabs defaultValue="hidden">
              <TabsList aria-label="Performance chart categories">
                <TabsTrigger value="channelcharts" className="flex items-center gap-2">
                  <BarChart3 size={16} aria-hidden="true" />
                  Channel Performance Charts
                </TabsTrigger>
                <TabsTrigger value="agegroupcharts" className="flex items-center gap-2">
                  <Users2 size={16} aria-hidden="true" />
                  Age Group Performance Charts
                </TabsTrigger>
                <TabsTrigger value="countrycharts" className="flex items-center gap-2">
                  <Globe size={16} aria-hidden="true" />
                  Country Performance Charts
                </TabsTrigger>
                <TabsTrigger value="hidden" className="flex items-center gap-2">
                  <EyeOff size={16} aria-hidden="true" />
                  Hide Tables
                </TabsTrigger>
              </TabsList>
              <TabsContent value="channelcharts">
                <ChannelPerformanceCharts />
              </TabsContent>
              <TabsContent value="agegroupcharts">
                <AgeGroupPerformanceCharts />
              </TabsContent>
              <TabsContent value="countrycharts">
                <CountryPerformanceCharts />
              </TabsContent>
              <TabsContent value="hidden"></TabsContent>
            </Tabs>
          </section>

          {/* Detailed Revenue & Ad Spend Chart */}
          <section aria-label="Revenue and ad spend analysis">
            <RevenueAdSpendChart />
          </section>

          <section className="grid grid-cols-2 gap-4" aria-label="Channel and cost metrics">
            <ChannelContributionChart />
            <CostMetricsHeatmap />
          </section>
        </div>
      </ProphetPredictionsProvider>
    </DatabaseOperationsProvider>
  );
}
