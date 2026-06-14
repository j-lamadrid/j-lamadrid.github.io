import React from "react";
import { FiExternalLink } from "react-icons/fi";
import favoriteAlbums from "./albums";

function formatReleaseYear(releaseDate) {
  return releaseDate ? releaseDate.slice(0, 4) : "----";
}

function formatAlbumRuntime(seconds) {
  const totalMinutes = Math.round(seconds / 60);
  const hours = Math.floor(totalMinutes / 60);
  const minutes = totalMinutes % 60;

  if (hours > 0) {
    return `${hours}h ${minutes}m`;
  }

  return `${minutes}m`;
}

function FavoriteAlbums() {
  return (
    <section className="favorite-albums-section reveal-up delay-2" aria-labelledby="favorite-albums-heading">
      <div className="favorite-albums-head">
        <div>
          <p className="section-kicker">Favorite Albums</p>
        </div>
      </div>

      <div className="favorite-albums-grid">
        {favoriteAlbums.map((album) => (
          <a
            aria-label={`Open ${album.title} by ${album.artist} on Deezer`}
            className="favorite-album-card"
            href={album.link}
            key={album.id}
            rel="noreferrer"
            target="_blank"
          >
            <img src={album.cover} alt={`${album.title} album cover`} loading="lazy" />
            <span className="favorite-album-scanline" aria-hidden="true" />
            <span className="favorite-album-overlay">
              <span className="favorite-album-meta">
                <span>{formatReleaseYear(album.releaseDate)}</span>
                <span>{album.tracks} tracks</span>
                <span>{formatAlbumRuntime(album.duration)}</span>
              </span>
              <span className="favorite-album-title">{album.title}</span>
              <span className="favorite-album-artist">{album.artist}</span>
              <span className="favorite-album-detail">{album.genres.join(" / ")}</span>
              <span className="favorite-album-detail">{album.label}</span>
              <span className="favorite-album-link">
                Deezer <FiExternalLink />
              </span>
            </span>
          </a>
        ))}
      </div>
    </section>
  );
}

export default FavoriteAlbums;
