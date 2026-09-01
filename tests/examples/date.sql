-- DROP DATABASE IF EXISTS instrument WITH (FORCE);
CREATE DATABASE date_tables WITH TEMPLATE template0 ENCODING = 'UTF8' ;
ALTER DATABASE date_tables OWNER TO postgres;

\connect date_tables

CREATE TABLE public.person (
    id INTEGER NOT NULL,
    name TEXT NOT NULL,
    timestamp_of_birth TIMESTAMP WITH TIME ZONE NOT NULL,
    date_of_birth DATE NOT NULL
);

ALTER TABLE ONLY public.person ADD CONSTRAINT person_pkey PRIMARY KEY (id);

ALTER TABLE public.person OWNER TO postgres;

INSERT INTO public.person VALUES (1, 'Bobby', '1991-01-08 11:12:13+00:00', '1991-01-08');
INSERT INTO public.person VALUES (2, 'Mary', '1989-03-04 20:19:18+00:00', '1989-03-04');

CREATE TABLE public.happening (
    id INTEGER NOT NULL,
    name TEXT NOT NULL,
    person_id INTEGER NOT NULL,
    at_time TIMESTAMP WITH TIME ZONE NOT NULL,
    at_date DATE NOT NULL
);

ALTER TABLE ONLY public.happening ADD CONSTRAINT happening_pkey PRIMARY KEY (id);
ALTER TABLE ONLY public.happening
    ADD CONSTRAINT person_id_fkey FOREIGN KEY (person_id) REFERENCES public.person(id);

ALTER TABLE public.happening OWNER TO postgres;

INSERT INTO public.happening VALUES (1, 'stepped on a tack', 1, '1997-04-20 04:05:06+00:00', '1997-04-20');
INSERT INTO public.happening VALUES (2, 'had a dream', 2, '1997-04-20 04:04:16+00:00', '1997-04-20');
INSERT INTO public.happening VALUES (3, 'kicked a can', 1, '2001-12-23 07:05:06+00:00', '2001-12-23');
INSERT INTO public.happening VALUES (4, 'steppen in gum', 2, '2003-11-03 04:15:26+00:00', '2003-11-03');
