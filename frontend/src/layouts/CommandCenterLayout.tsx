import React from 'react';

interface LayoutProps {
  children: React.ReactNode;
}

export const CommandCenterLayout: React.FC<LayoutProps> = ({ children }) => {
  return (
    <div className="min-h-screen flex flex-col bg-[#09090b]">
      {/* Dynamic Sci-Fi background element */}
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top_right,rgba(29,78,216,0.06),transparent_50%)] pointer-events-none" />
      
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8 relative z-10">
        {children}
      </main>
      
      <footer className="py-6 border-t border-zinc-900 bg-zinc-950/20 text-center relative z-10">
        <p className="text-xs text-zinc-600 font-semibold tracking-wider uppercase">
          Eagle Eye Smart Surveillance Dashboard • Secured Command Node
        </p>
      </footer>
    </div>
  );
};
export default CommandCenterLayout;
