"use client";

import { ArrowRight } from "lucide-react"
import ColorBends from './components/ColorBends';
import TextType from './components/TextType';
import Link from 'next/link';
import { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Plus_Jakarta_Sans } from 'next/font/google';

const plusJakartaSans = Plus_Jakarta_Sans({ subsets: ['latin'] });

export default function Home() {
  const [showTitle, setShowTitle] = useState(false);
  const [showSubtitle, setShowSubtitle] = useState(false);
  const [showButton, setShowButton] = useState(false);

  useEffect(() => {
    const titleTimer = setTimeout(() => setShowTitle(true), 750);
    const subtitleTimer = setTimeout(() => setShowSubtitle(true), 750);
    const buttonTimer = setTimeout(() => setShowButton(true), 1250);

    return () => {
      clearTimeout(titleTimer);
      clearTimeout(subtitleTimer);
      clearTimeout(buttonTimer);
    };
  }, []);

  return (
    <main className="relative flex min-h-screen w-full items-center justify-center font-sans bg-black dark:bg-black">
      <div className="w-full h-screen">
        <ColorBends
        rotation={180}
        frequency={1}
        transparent={true}
        warpStrength={0}
        mouseInfluence={0}/>
      </div>
      
      {/* GitHub Logo */}
      <a 
        href="https://github.com/tshrrx/InvenSight"
        target="_blank"
        rel="noopener noreferrer"
        className="absolute top-8 right-8 z-20 p-3 rounded-full text-white hover:text-gray-300 transition-all duration-300 ring-2 ring-offset-2 ring-offset-black animate-[pulse_3s_ease-in-out_infinite]"
        style={{
          boxShadow: '0 0 20px rgba(139, 92, 246, 0.5), 0 0 40px rgba(59, 130, 246, 0.3)',
          background: 'linear-gradient(45deg, rgba(139, 92, 246, 0.1), rgba(59, 130, 246, 0.1))',
        }}
      >
        <svg className="w-10 h-10" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true">
          <path fillRule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" clipRule="evenodd" />
        </svg>
      </a>

      <div className="absolute inset-0 bg-transparent z-10">
        <div className="flex flex-col items-center justify-center h-full gap-4">
          <h1
            className={`${plusJakartaSans.className} text-8xl font-bold text-white drop-shadow-[0_10px_30px_rgba(0,0,0,0.9)] transition-opacity duration-1000 ${
              showTitle ? 'opacity-100' : 'opacity-0'
            }`}
          >
            InvenSight.
          </h1>
          <div
            className={`text-2xl text-white drop-shadow-[0_10px_30px_rgba(0,0,0,0.9)] transition-opacity duration-1000 ${
              showSubtitle ? 'opacity-100' : 'opacity-0'
            }`}
          >
            {showSubtitle && (
              <TextType
                text={["Transform your retail data into instant insights."]}
                typingSpeed={50}
                loop={false}
                showCursor={false}
              />
            )}
          </div>
          <Link href="/main" className="mt-8">
            <Button
              size="icon"
              variant="outline"
              className={`rounded-full w-32 h-12 transition-all duration-1000 ${
                showButton ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'
              }`}
            >
              Try Now <ArrowRight className="w-12 h-12" />
            </Button>
          </Link>
        </div>
      </div>
    </main>
  );
}