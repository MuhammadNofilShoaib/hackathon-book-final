import React from 'react';
import clsx from 'clsx';
import styles from './HomepageFeatures.module.css';

const FeatureList = [
  {
    title: 'Physical AI & Humanoid Robotics Textbook',
    description: (
      <>
        A comprehensive textbook covering all aspects of Physical AI and Humanoid Robotics,
        from basic concepts to advanced implementations.
      </>
    ),
  },
  {
    title: 'Personalized Learning Paths',
    description: (
      <>
        Content tailored to your skill level - Beginner, Intermediate, or Advanced.
        Choose the path that best matches your experience.
      </>
    ),
  },
  {
    title: 'Multilingual Support',
    description: (
      <>
        Available in multiple languages including English and Urdu to support
        diverse learning communities.
      </>
    ),
  },
];

function Feature({title, description}) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center padding-horiz--md">
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}